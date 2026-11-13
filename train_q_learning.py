"""Huấn luyện và đánh giá Tabular Q-learning theo RL Algorithm Development & Evaluation Standard v1.0."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dispatch_env import (
    AmbulanceDispatchEnv,
    AmbulanceStatus,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_MAX_TIME,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_SEED,
    REWARD_CONFIG,
)
from q_learning import QLearningAgent
from ambulance_q_learning_adapter import AmbulanceDispatchAdapter


HIGH_SEVERITY_LEVEL = 3  # Định nghĩa thống nhất V-03: mức độ nghiêm trọng cao nhất (deadline = 30 phút).


def _count_priority_violation(
    adapter: AmbulanceDispatchAdapter,
    obs: Mapping[str, Any],
    dispatched_pairs: List[Tuple[int, int]],
) -> int:
    """Đếm vi phạm ưu tiên tại bước quyết định (cùng định nghĩa cho Q-learning và Auto-dispatch).

    Vi phạm xảy ra khi có ít nhất 1 cuộc gọi khả thi mức cao hơn (ETA <= deadline - t)
    bị bỏ lại chưa phân công trong khi một xe được giao cho cuộc gọi mức thấp hơn
    vẫn có thể tới kịp cuộc gọi mức cao hơn đó.
    """
    if not dispatched_pairs:
        return 0

    pending_calls = [c for c in obs["call_slots"] if c is not None]
    if len(pending_calls) < 2:
        return 0

    current_minute = float(obs["current_minute"])
    assigned_call_ids = {int(cid) for _, cid in dispatched_pairs}
    call_by_id = {int(c["call_id"]): c for c in pending_calls}
    idle_ambs = adapter.get_idle_ambulances_from_obs(obs)
    if not idle_ambs:
        return 0

    # Các cuộc gọi khả thi bị bỏ lại chưa được giao xe ở bước này.
    unassigned_feasible = []
    for call_dict in pending_calls:
        cid = int(call_dict["call_id"])
        if cid in assigned_call_ids:
            continue
        rem = max(0.0, float(call_dict["deadline"]) - current_minute)
        best_eta = min(
            adapter.compute_eta_from_obs(
                obs,
                int(amb["current_edge_id"]),
                float(amb["distance_from_start"]),
                int(call_dict["edge_id"]),
                float(call_dict["dist"]),
            )
            for amb in idle_ambs
        )
        if best_eta <= rem:
            unassigned_feasible.append(call_dict)

    if not unassigned_feasible:
        return 0

    ambs_by_id = {int(a["id"]): a for a in idle_ambs}
    for amb_id, cid in dispatched_pairs:
        low_call = call_by_id.get(int(cid))
        amb = ambs_by_id.get(int(amb_id))
        if low_call is None or amb is None:
            continue
        for high_call in unassigned_feasible:
            if int(high_call["severity"]) > int(low_call["severity"]):
                rem_high = max(0.0, float(high_call["deadline"]) - current_minute)
                eta_to_high = adapter.compute_eta_from_obs(
                    obs,
                    int(amb["current_edge_id"]),
                    float(amb["distance_from_start"]),
                    int(high_call["edge_id"]),
                    float(high_call["dist"]),
                )
                if eta_to_high <= rem_high:
                    return 1
    return 0


def run_episode(
    env: AmbulanceDispatchEnv,
    adapter: AmbulanceDispatchAdapter,
    agent: Optional[QLearningAgent],
    training: bool,
    seed: int,
    capture_log: bool = False,
) -> Dict[str, object]:
    """Chạy một lượt mô phỏng từ đầu đến khi môi trường kết thúc hoặc truncated."""
    obs = env.reset(seed=seed)
    t0_reset_reward = float(obs.get("last_step_reward", 0.0))
    total_reward = t0_reset_reward
    log_lines: List[str] = []
    done = False

    decision_steps = 0
    decision_noop = 0
    total_steps = 0
    total_noop = 0
    state_hits = 0
    mask_violations = 0

    wait_penalty_sum = 0.0
    failure_penalty_sum = 0.0
    distance_penalty_sum = 0.0
    completion_bonus_sum = 0.0

    slack_values: List[float] = []
    priority_violations = 0
    busy_amb_minutes = 0
    action_trace: List[int] = []

    seen_states = {state for (state, _) in agent.q_table} if agent is not None else set()
    pending_transition: Optional[Tuple[str, int, int]] = None
    macro_reward = 0.0
    macro_steps = 0
    last_info: Dict[str, Any] = {}

    while not done:
        total_steps += 1
        pre_obs = obs
        is_decision = adapter.is_decision_state_from_obs(pre_obs)
        if is_decision:
            decision_steps += 1

        # Đếm số xe đang bận ở đầu bước để tính Ambulance Utilization thống nhất.
        busy_amb_minutes += sum(
            1
            for amb in pre_obs["ambulances"].values()
            if amb["status"] != AmbulanceStatus.IDLE.value
        )

        if agent is None:
            obs, reward, done, info = env.step(auto_dispatch=True)
            action_id = int(info["action_id"])
            if not bool(pre_obs["action_mask"][action_id]):
                mask_violations += 1
            if is_decision and action_id == 0:
                decision_noop += 1
            if action_id == 0:
                total_noop += 1
            action_text = "Heuristic auto_dispatch"
        else:
            if is_decision:
                state = adapter.encode_state(env, obs=pre_obs)
                candidates = adapter.candidate_actions(env, obs=pre_obs)
                action_keys = adapter.get_action_keys(env, candidates, obs=pre_obs)

                if state in seen_states:
                    state_hits += 1

                if training and pending_transition is not None:
                    prev_state, prev_action, prev_key = pending_transition
                    agent.update(
                        state=prev_state,
                        action=prev_action,
                        reward=macro_reward,
                        next_state=state,
                        next_valid_actions=candidates,
                        done=False,
                        steps=macro_steps,
                        action_key=prev_key,
                        next_action_keys=action_keys,
                    )
                    seen_states.add(prev_state)

                action_id = agent.choose_action(
                    state, candidates, explore=training, action_keys=action_keys
                )
                if not bool(pre_obs["action_mask"][action_id]):
                    mask_violations += 1
                chosen_key = action_keys[action_id]
                if action_id == 0:
                    decision_noop += 1
                    total_noop += 1

                decoded_action = env.decode_action(action_id)
                action_text = f"action_id={action_id} (key={chosen_key}), phân công={decoded_action}"

                # Gán tín chỉ hoàn thành sớm cho các ca khả thi được phân công để giải quyết bài toán credit assignment trễ.
                dispatch_credit = 0.0
                if training and action_id != 0:
                    call_by_id = {
                        int(c["call_id"]): c
                        for c in pre_obs["call_slots"]
                        if c is not None
                    }
                    ambs_by_id = {int(aid): a for aid, a in pre_obs["ambulances"].items()}
                    t_disp = float(pre_obs["current_minute"])
                    for amb_id, call_id in decoded_action:
                        c_dict = call_by_id.get(int(call_id))
                        a_dict = ambs_by_id.get(int(amb_id))
                        if c_dict is not None and a_dict is not None:
                            eta = adapter.compute_eta_from_obs(
                                pre_obs,
                                int(a_dict["current_edge_id"]),
                                float(a_dict["distance_from_start"]),
                                int(c_dict["edge_id"]),
                                float(c_dict["dist"]),
                            )
                            rem = max(0.0, float(c_dict["deadline"]) - t_disp)
                            if eta <= rem:
                                sev = int(c_dict["severity"])
                                dispatch_credit += REWARD_CONFIG[sev]["complete_reward"]

                obs, reward, done, info = adapter.step(env, action_id)
                pending_transition = (state, action_id, chosen_key)
                
                # Trong huấn luyện: trừ completion_bonus trễ để không gán trùng hoặc gán nhầm cho action khác
                bd = info.get("reward_breakdown", {})
                step_comp = float(bd.get("completion_bonus", 0.0))
                shaped_r = reward - step_comp + dispatch_credit if training else reward
                macro_reward = shaped_r
                macro_steps = 1
            else:
                action_id = 0
                if not bool(pre_obs["action_mask"][action_id]):
                    mask_violations += 1
                total_noop += 1
                action_text = "action_id=0 (WAIT - không có quyết định)"
                obs, reward, done, info = adapter.step(env, 0)
                if pending_transition is not None:
                    bd = info.get("reward_breakdown", {})
                    step_comp = float(bd.get("completion_bonus", 0.0))
                    shaped_r = reward - step_comp if training else reward
                    macro_reward += (agent.gamma ** macro_steps) * shaped_r
                    macro_steps += 1

            if done and training and pending_transition is not None:
                prev_state, prev_action, prev_key = pending_transition
                agent.update(
                    state=prev_state,
                    action=prev_action,
                    reward=macro_reward,
                    next_state="TERMINAL",
                    next_valid_actions=[],
                    done=True,
                    steps=macro_steps,
                    action_key=prev_key,
                )
                seen_states.add(prev_state)
                pending_transition = None

        action_trace.append(action_id)
        last_info = info

        # Ghi nhận Slack tại thời điểm điều phối & Priority Violation từ pre_obs + info["decoded_action"].
        dispatched_pairs = list(info.get("decoded_action", []))
        if is_decision and dispatched_pairs:
            call_by_id = {
                int(c["call_id"]): c
                for c in pre_obs["call_slots"]
                if c is not None
            }
            ambs_by_id = {int(aid): a for aid, a in pre_obs["ambulances"].items()}
            t_disp = float(pre_obs["current_minute"])
            for amb_id, call_id in dispatched_pairs:
                c_dict = call_by_id.get(int(call_id))
                a_dict = ambs_by_id.get(int(amb_id))
                if c_dict is not None and a_dict is not None:
                    eta = adapter.compute_eta_from_obs(
                        pre_obs,
                        int(a_dict["current_edge_id"]),
                        float(a_dict["distance_from_start"]),
                        int(c_dict["edge_id"]),
                        float(c_dict["dist"]),
                    )
                    slack_values.append(float(c_dict["deadline"]) - (t_disp + eta))
            priority_violations += _count_priority_violation(adapter, pre_obs, dispatched_pairs)

        breakdown = info.get("reward_breakdown", {})
        wait_penalty_sum += float(breakdown.get("wait_penalty", 0.0))
        failure_penalty_sum += float(breakdown.get("failure_penalty", 0.0))
        distance_penalty_sum += float(breakdown.get("distance_penalty", 0.0))
        completion_bonus_sum += float(breakdown.get("completion_bonus", 0.0))

        total_reward += reward
        if capture_log:
            log_lines.append(f"Quyết định: {action_text}\n{env.format_log_step()}")

    if training and agent is not None:
        agent.replay_experiences()

    # Đọc read-only lịch sử env.calls theo Mục 6.2 #5 để phân tách KPI theo severity & Unresolved.
    calls = list(env.calls.values())
    call_count = len(calls)
    served_count = int(env.served_calls_count)
    failed_count = int(env.failed_calls_count)

    # Unresolved (Mục 6.2 #2): chỉ tính pending + waiting + at_scene (không tính en_route_hospital/at_hospital).
    if bool(last_info.get("truncated")) and isinstance(last_info.get("backlog"), dict):
        backlog = last_info["backlog"]
        unresolved_count = (
            len(backlog.get("pending_calls", []))
            + len(backlog.get("waiting_calls", []))
            + len(backlog.get("at_scene_calls", []))
        )
    else:
        unresolved_count = sum(
            1
            for call in calls
            if call.status.value in ("Yêu cầu xử lý", "Đợi xe cứu thương", "Đang xử lý")
        )

    high_calls = [c for c in calls if int(c.severity) == HIGH_SEVERITY_LEVEL]
    high_served = sum(1 for c in high_calls if c.status.value == "Hoàn thành")
    high_failed = sum(1 for c in high_calls if c.status.value == "Thất bại")
    high_responses = [
        float(c.arrived_at_scene_time) - float(c.created_time)
        for c in high_calls
        if c.arrived_at_scene_time is not None
    ]

    avg_wait_time = (sum(float(c.wait_time) for c in calls) / call_count) if call_count > 0 else None
    total_distance = sum(
        float(amb["total_distance_traveled"]) for amb in obs["ambulances"].values()
    )
    num_ambs = max(1, len(obs["ambulances"]))
    amb_utilization = busy_amb_minutes / max(1, num_ambs * total_steps)
    reward_reconciled = abs(total_reward - float(obs["total_reward"])) < 1e-6

    return {
        "seed": seed,
        "total_reward": total_reward,
        "t0_reset_reward": t0_reset_reward,
        "reward_reconciled": reward_reconciled,
        "wait_penalty": wait_penalty_sum,
        "failure_penalty": failure_penalty_sum,
        "distance_penalty": distance_penalty_sum,
        "completion_bonus": completion_bonus_sum,
        "calls": call_count,
        "served": served_count,
        "failed": failed_count,
        "unresolved": unresolved_count,
        "perfect": int(env.perfect_calls_count),
        "success_rate": (served_count / call_count) if call_count > 0 else None,
        "failure_rate": (failed_count / call_count) if call_count > 0 else None,
        "unresolved_rate": (unresolved_count / call_count) if call_count > 0 else None,
        "perfect_rate": (int(env.perfect_calls_count) / call_count) if call_count > 0 else None,
        "high_sev_calls": len(high_calls),
        "high_sev_served": high_served,
        "high_sev_failed": high_failed,
        "high_sev_response_count": len(high_responses),
        "high_sev_served_rate": (high_served / len(high_calls)) if high_calls else None,
        "high_sev_failed_rate": (high_failed / len(high_calls)) if high_calls else None,
        "high_sev_response_time": (
            sum(high_responses) / len(high_responses) if high_responses else None
        ),
        "avg_wait_time": avg_wait_time,
        "avg_slack_at_dispatch": (
            sum(slack_values) / len(slack_values) if slack_values else None
        ),
        "priority_violations": priority_violations,
        "priority_violation_rate": (priority_violations / decision_steps) if decision_steps > 0 else None,
        "total_distance": total_distance,
        "ambulance_utilization": amb_utilization,
        "decision_steps": decision_steps,
        "decision_noop_rate": (decision_noop / decision_steps) if decision_steps > 0 else None,
        "total_noop_rate": total_noop / max(1, total_steps),
        "state_hit_rate": (state_hits / decision_steps) if decision_steps > 0 else None,
        "mask_violations": mask_violations,
        "steps": int(obs["current_minute"]),
        "truncated": bool(last_info.get("truncated", False)),
        "action_trace": action_trace,
        "log": "\n\n".join(log_lines),
    }


def evaluate_policy(
    make_env: Callable[[int], AmbulanceDispatchEnv],
    adapter: AmbulanceDispatchAdapter,
    agent: QLearningAgent,
    seeds: List[int],
) -> Tuple[float, float]:
    """Đánh giá nhanh chính sách hiện tại (explore=False) trên tập validation seeds."""
    results = [
        run_episode(make_env(seed), adapter, agent, training=False, seed=seed)
        for seed in seeds
    ]
    valid_succ = [float(row["success_rate"]) for row in results if row["success_rate"] is not None]
    avg_success = sum(valid_succ) / len(valid_succ) if valid_succ else 0.0
    avg_reward = sum(float(row["total_reward"]) for row in results) / len(results)
    return avg_success, avg_reward


def verify_correctness_gate(
    make_env: Callable[[int], AmbulanceDispatchEnv],
    adapter: AmbulanceDispatchAdapter,
    agent: QLearningAgent,
    evaluation_rows: List[Dict[str, object]],
    q_table_before_eval: Dict[Tuple[str, int], float],
    visits_before_eval: Dict[Tuple[str, int], int],
    repro_seed: int,
) -> Dict[str, Tuple[str, str]]:
    """Kiểm tra Correctness Gate theo Mục 7 của RL Standard v1.0.

    Trả về ``{check_name: (status, detail)}`` với ``status`` thuộc ``PASS``, ``FAIL``, ``UNVERIFIED``.
    """
    q_rows = [row for row in evaluation_rows if row["strategy"] == "Q-learning"]

    # G1: Observation keys & enum mapping
    sample_env = make_env(repro_seed)
    sample_obs = sample_env.reset(seed=repro_seed)
    required_keys = {"current_minute", "call_slots", "action_mask", "ambulances", "edges_by_id", "total_reward"}
    g1_obs_ok = required_keys.issubset(sample_obs.keys()) and (
        AmbulanceStatus.IDLE.value == adapter.IDLE_STATUS_STR
    )

    # G2: Action Mask compliance across all evaluation episodes
    total_mask_violations = sum(int(row["mask_violations"]) for row in evaluation_rows)
    g2_mask_ok = total_mask_violations == 0

    # G3: Learning Update Correctness (Bellman Semi-MDP, Terminal, adaptive_alpha, 1-ep probe update & finite Q)
    math_agent = QLearningAgent(
        alpha=0.5,
        gamma=0.9,
        adaptive_alpha=True,
        alpha_min=0.1,
        alpha_decay=1.0,
        seed=1,
    )
    math_agent.q_table[("s2", 1)] = 10.0
    math_agent.update("s1", 1, 2.0, "s2", [1], done=False, steps=3)
    expected_q1 = 0.5 * (2.0 + (0.9 ** 3) * 10.0)
    g3_semimdp_ok = abs(math_agent.q_table[("s1", 1)] - expected_q1) < 1e-9

    math_agent.update("s1", 1, 2.0, "s2", [1], done=False, steps=3)
    expected_q2 = expected_q1 + 0.25 * (2.0 + (0.9 ** 3) * 10.0 - expected_q1)
    g3_alpha_ok = abs(math_agent.q_table[("s1", 1)] - expected_q2) < 1e-9

    math_agent.update("s_term", 1, 4.0, "s2", [1], done=True, steps=2)
    g3_terminal_ok = abs(math_agent.q_table[("s_term", 1)] - 2.0) < 1e-9

    probe_agent = QLearningAgent(
        alpha=agent.alpha,
        gamma=agent.gamma,
        epsilon=agent.epsilon,
        epsilon_min=agent.epsilon_min,
        epsilon_decay=agent.epsilon_decay,
        seed=repro_seed,
        adaptive_alpha=agent.adaptive_alpha,
        alpha_min=agent.alpha_min,
        alpha_decay=agent.alpha_decay,
        initial_q=agent.initial_q,
        replay_capacity=agent.replay_capacity,
        replay_batch_size=agent.replay_batch_size,
    )
    probe_result = run_episode(
        make_env(repro_seed), adapter, probe_agent, training=True, seed=repro_seed
    )
    probe_has_decisions = int(probe_result["decision_steps"]) > 0
    g3_probe_updated = (len(probe_agent.q_table) > 0) if probe_has_decisions else True
    g3_probe_visits = (sum(probe_agent.visit_counts.values()) > 0) if probe_has_decisions else True
    g3_finite_ok = all(math.isfinite(v) for v in probe_agent.q_table.values()) and all(
        math.isfinite(v) for v in agent.q_table.values()
    )
    g3_ok = g3_semimdp_ok and g3_alpha_ok and g3_terminal_ok and g3_probe_updated and g3_probe_visits and g3_finite_ok

    # G4: Reward reconciliation (t=0 + sum(step_rewards) == env.total_reward) & call accounting
    g4_reward_ok = all(bool(row["reward_reconciled"]) for row in evaluation_rows)
    accounting_ok = all(
        int(row["served"]) + int(row["failed"]) + int(row["unresolved"]) == int(row["calls"])
        for row in evaluation_rows
    )

    # G5: Evaluation does not mutate Q-table or visit_counts
    g5_no_eval_update = (agent.q_table == q_table_before_eval) and (
        agent.visit_counts == visits_before_eval
    )

    # G6: Environment Integrity (AST static check + default config + git clean check nếu khả dụng)
    project_root = Path(__file__).resolve().parent
    rl_files = ("q_learning.py", "ambulance_q_learning_adapter.py", "train_q_learning.py")
    env_mutations: List[str] = []
    q_core_env_imports: List[str] = []
    ast_checked = True
    try:
        for fname in rl_files:
            src = (project_root / fname).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if fname == "q_learning.py":
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split(".", 1)[0] in {"dispatch_env", "map_env"}:
                                q_core_env_imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.split(".", 1)[0] in {"dispatch_env", "map_env"}:
                            q_core_env_imports.append(node.module)
                if isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for tgt in targets:
                        if (
                            isinstance(tgt, (ast.Attribute, ast.Subscript))
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id in {"env", "obs", "city_map"}
                        ):
                            env_mutations.append(f"{fname}:{node.lineno}")
    except OSError:
        ast_checked = False

    g6_config_ok = (
        sample_env.num_ambulances == DEFAULT_NUM_AMBULANCES
        and sample_env.max_time == DEFAULT_MAX_TIME
        and sample_env.max_overtime == DEFAULT_MAX_OVERTIME
        and abs(float(sample_env.poisson_lambda) - float(DEFAULT_POISSON_LAMBDA)) < 1e-12
    )
    g6_static_ok = ast_checked and not env_mutations and not q_core_env_imports and g6_config_ok

    g6_git_status = "UNVERIFIED"
    g6_git_detail = "giới hạn: không kiểm được git diff cho dispatch_env/ & map_env/"
    try:
        git_proc = subprocess.run(
            ["git", "status", "--porcelain", "--", "dispatch_env/", "map_env/"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_proc.returncode == 0:
            dirty_lines = [
                line
                for line in git_proc.stdout.splitlines()
                if line.strip() and "__pycache__" not in line
            ]
            if dirty_lines:
                g6_git_status = "FAIL"
                g6_git_detail = f"phát hiện thay đổi chưa commit: {dirty_lines[:3]}"
            else:
                g6_git_status = "PASS"
                g6_git_detail = "sạch so với HEAD (giới hạn: không kiểm được thay đổi đã commit trước)"
        else:
            g6_git_detail = f"UNVERIFIED: git trả mã lỗi {git_proc.returncode} (không có git repo)"
    except FileNotFoundError:
        g6_git_detail = "UNVERIFIED: hệ thống không có lệnh git"

    # G7: Reproducibility on identical seed
    run_a = run_episode(make_env(repro_seed), adapter, agent, training=False, seed=repro_seed)
    run_b = run_episode(make_env(repro_seed), adapter, agent, training=False, seed=repro_seed)
    g7_reproducible = (
        abs(float(run_a["total_reward"]) - float(run_b["total_reward"])) < 1e-9
        and int(run_a["served"]) == int(run_b["served"])
        and int(run_a["failed"]) == int(run_b["failed"])
        and run_a["action_trace"] == run_b["action_trace"]
    )

    checks: Dict[str, Tuple[str, str]] = {
        "G1_observation_and_status_mapping": (
            "PASS" if g1_obs_ok else "FAIL",
            "Observation đủ khóa chuẩn và khớp AmbulanceStatus.IDLE",
        ),
        "G2_action_mask_zero_violations": (
            "PASS" if g2_mask_ok else "FAIL",
            f"mask_violations={total_mask_violations}",
        ),
        "G3_learning_update_correctness": (
            "PASS" if g3_ok else "FAIL",
            f"Semi-MDP/Terminal/adaptive_alpha + 1-ep probe Q={len(probe_agent.q_table)} cặp, finite={g3_finite_ok}",
        ),
        "G4_reward_reconciliation_with_t0": (
            "PASS" if (g4_reward_ok and accounting_ok) else "FAIL",
            "t0 + sum(step_rewards) == total_reward & served+failed+unresolved == calls",
        ),
        "G5_eval_policy_frozen": (
            "PASS" if g5_no_eval_update else "FAIL",
            "Q-table và visit_counts bất biến trong đánh giá",
        ),
        "G6_env_static_and_config": (
            "PASS" if g6_static_ok else "FAIL",
            f"AST mutations={len(env_mutations)}, core_env_imports={len(q_core_env_imports)}, default_config={g6_config_ok}",
        ),
        "G6_env_git_working_tree": (
            g6_git_status,
            g6_git_detail,
        ),
        "G7_seed_reproducibility": (
            "PASS" if (g7_reproducible and bool(q_rows)) else "FAIL",
            f"Lặp lại seed={repro_seed} khớp reward/served/failed/action_trace",
        ),
    }
    return checks


def print_episode_progress(
    phase: str,
    episode: int,
    total_episodes: int,
    started_at: float,
    result: Dict[str, object],
    details: str = "",
) -> None:
    """In một dòng tiến độ có cùng bố cục cho huấn luyện và đánh giá."""
    elapsed = time.perf_counter() - started_at
    average_seconds = elapsed / episode
    remaining_seconds = average_seconds * (total_episodes - episode)
    calls = int(result["calls"])
    served = int(result["served"])
    failed = int(result["failed"])
    succ_val = result["success_rate"]
    wait_val = result["avg_wait_time"]
    succ_str = f"{100 * float(succ_val):5.1f}%" if succ_val is not None else "  N/A "
    wait_str = f"{float(wait_val):4.1f}p" if wait_val is not None else " N/A "
    progress = 100 * episode / total_episodes

    message = (
        f"[{phase:<9}] {episode:>3}/{total_episodes:<3} ({progress:5.1f}%) | "
        f"Reward {float(result['total_reward']):>9.2f} | "
        f"Cuộc gọi {served:>3}/{calls:<3} hoàn thành ({succ_str}), "
        f"thất bại {failed:>3}, chờ TB {wait_str} | "
        f"Đã chạy {elapsed / 60:5.1f}p, còn ~{remaining_seconds / 60:5.1f}p"
    )
    if details:
        message += f" | {details}"

    print(message, flush=True)


def _stats(
    values: Sequence[Optional[object]], scale: float = 1.0
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """Tính Mean, Std, Min, Max và số mẫu hữu hiệu (loại bỏ None và NaN)."""
    valid: List[float] = []
    for raw in values:
        if raw is None or raw == "":
            continue
        fv = float(raw)
        if math.isnan(fv):
            continue
        valid.append(fv * scale)
    if not valid:
        return None, None, None, None, 0
    mean_v = statistics.mean(valid)
    std_v = statistics.stdev(valid) if len(valid) > 1 else 0.0
    return mean_v, std_v, min(valid), max(valid), len(valid)


def print_final_summary(
    strategy: str,
    rows: List[Dict[str, object]],
) -> None:
    """In Mean, Std, Min, Max kèm số lượng mẫu thực tế của từng KPI theo chuẩn báo cáo Mục 9."""
    metrics = [
        ("Served Rate (%)", "success_rate", 100.0),
        ("Failed Rate (%)", "failure_rate", 100.0),
        ("High-Sev Served (%)", "high_sev_served_rate", 100.0),
        ("High-Sev Resp (p)", "high_sev_response_time", 1.0),
        ("Avg Wait (p)", "avg_wait_time", 1.0),
        ("Avg Slack (p)", "avg_slack_at_dispatch", 1.0),
        ("Prio Violation (%)", "priority_violation_rate", 100.0),
        ("Episode Reward", "total_reward", 1.0),
        ("Total Distance (km)", "total_distance", 1.0),
        ("Utilization (%)", "ambulance_utilization", 100.0),
        ("No-op QĐ (%)", "decision_noop_rate", 100.0),
        ("State Hit (%)", "state_hit_rate", 100.0),
    ]
    total_episodes = len(rows)
    total_high_calls = sum(int(r.get("high_sev_calls", 0)) for r in rows)
    total_high_reached = sum(int(r.get("high_sev_response_count", 0)) for r in rows)
    total_high_served = sum(int(r.get("high_sev_served", 0)) for r in rows)
    print(
        f"\n  [{strategy}] (N={total_episodes} evaluation seeds | "
        f"Mức {HIGH_SEVERITY_LEVEL}: {total_high_calls} ca, tiếp cận {total_high_reached}, hoàn thành {total_high_served})"
    )
    for label, key, scale in metrics:
        mean_v, std_v, min_v, max_v, n_valid = _stats([r.get(key) for r in rows], scale=scale)
        if mean_v is None or std_v is None or min_v is None or max_v is None:
            print(f"    {label:<21}:      N/A ±    N/A  (min=     N/A, max=     N/A, n=0/{total_episodes})")
        else:
            print(
                f"    {label:<21}: {mean_v:8.2f} ± {std_v:6.2f}  "
                f"(min={min_v:8.2f}, max={max_v:8.2f}, n={n_valid}/{total_episodes})"
            )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Ghi metrics thành CSV; bỏ các trường log/trace dài khỏi bảng chỉ số."""
    if not rows:
        raise ValueError("Không có kết quả để ghi CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = {"log", "action_trace"}
    columns = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def plot_results(
    training_rows: List[Dict[str, object]],
    evaluation_rows: List[Dict[str, object]],
    path: Path,
) -> None:
    """Vẽ reward, tỷ lệ hoàn thành trong huấn luyện và so sánh đánh giá."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    rewards = [float(row["total_reward"]) for row in training_rows]
    window = min(15, max(1, len(rewards) // 5))
    reward_avg = [
        sum(rewards[max(0, i - window + 1):i + 1]) / len(rewards[max(0, i - window + 1):i + 1])
        for i in range(len(rewards))
    ]
    axes[0].plot(range(1, len(rewards) + 1), rewards, alpha=0.3, label="Reward mỗi episode")
    axes[0].plot(range(1, len(reward_avg) + 1), reward_avg, linewidth=2, label=f"TB trượt ({window})")
    axes[0].set(title="Reward trong huấn luyện", xlabel="Episode", ylabel="Tổng reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    successes = [
        100 * float(row["success_rate"]) if row["success_rate"] is not None else 0.0
        for row in training_rows
    ]
    succ_avg = [
        sum(successes[max(0, i - window + 1):i + 1]) / len(successes[max(0, i - window + 1):i + 1])
        for i in range(len(successes))
    ]
    axes[1].plot(range(1, len(successes) + 1), successes, alpha=0.3, label="Hoàn thành mỗi episode")
    axes[1].plot(range(1, len(succ_avg) + 1), succ_avg, linewidth=2, color="#d95f02", label=f"TB trượt ({window})")
    axes[1].set(ylim=(0, 100), title="Tỷ lệ hoàn thành trong huấn luyện", xlabel="Episode", ylabel="Phần trăm (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    strategies = ("Q-learning", "Auto-dispatch")
    means = []
    for strategy in strategies:
        rates = [
            100 * float(row["success_rate"])
            for row in evaluation_rows
            if row["strategy"] == strategy and row["success_rate"] is not None
        ]
        means.append(sum(rates) / max(1, len(rates)))
    axes[2].bar([0, 1], means, color=["#2878b5", "#888888"])
    axes[2].set_xticks([0, 1], strategies)
    axes[2].set(ylim=(0, 100), title="Đánh giá trên seed mới", ylabel="Hoàn thành (%)")
    axes[2].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_summary_csv(path: Path, evaluation_rows: List[Dict[str, object]]) -> None:
    """Ghi bảng tổng kết Mean, Std, Min, Max, n_valid của 2 chiến lược ra file CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("Served Rate (%)", "success_rate", 100.0),
        ("Failed Rate (%)", "failure_rate", 100.0),
        ("Unresolved Rate (%)", "unresolved_rate", 100.0),
        ("High-Sev Served (%)", "high_sev_served_rate", 100.0),
        ("High-Sev Failed (%)", "high_sev_failed_rate", 100.0),
        ("High-Sev Resp (p)", "high_sev_response_time", 1.0),
        ("Avg Wait (p)", "avg_wait_time", 1.0),
        ("Avg Slack (p)", "avg_slack_at_dispatch", 1.0),
        ("Prio Violation (%)", "priority_violation_rate", 100.0),
        ("Episode Reward", "total_reward", 1.0),
        ("Total Distance (km)", "total_distance", 1.0),
        ("Utilization (%)", "ambulance_utilization", 100.0),
        ("No-op QĐ (%)", "decision_noop_rate", 100.0),
        ("State Hit (%)", "state_hit_rate", 100.0),
    ]
    rows = []
    strategies = ("Q-learning", "Auto-dispatch")
    for strat in strategies:
        strat_rows = [r for r in evaluation_rows if r["strategy"] == strat]
        n_total = len(strat_rows)
        for label, key, scale in metrics:
            mean_v, std_v, min_v, max_v, n_valid = _stats([r.get(key) for r in strat_rows], scale=scale)
            rows.append({
                "Strategy": strat,
                "Metric": label,
                "Mean": round(mean_v, 2) if mean_v is not None else "N/A",
                "Std": round(std_v, 2) if std_v is not None else "N/A",
                "Min": round(min_v, 2) if min_v is not None else "N/A",
                "Max": round(max_v, 2) if max_v is not None else "N/A",
                "N_Valid": n_valid,
                "Total_Episodes": n_total,
            })
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Strategy", "Metric", "Mean", "Std", "Min", "Max", "N_Valid", "Total_Episodes"])
        writer.writeheader()
        writer.writerows(rows)


def plot_evaluation_comparison(evaluation_rows: List[Dict[str, object]], path: Path) -> None:
    """Vẽ biểu đồ so sánh chuyên sâu 6 chỉ số chính giữa Q-learning và Auto-dispatch."""
    kpis = [
        ("Served Rate (%)", "success_rate", 100.0, (0, 100)),
        ("High-Sev Served (%)", "high_sev_served_rate", 100.0, (0, 100)),
        ("Avg Wait Time (p)", "avg_wait_time", 1.0, None),
        ("Episode Reward", "total_reward", 1.0, None),
        ("Utilization (%)", "ambulance_utilization", 100.0, (0, 100)),
        ("No-op QĐ (%)", "decision_noop_rate", 100.0, (0, 100)),
    ]
    strategies = ("Q-learning", "Auto-dispatch")
    colors = ["#1f77b4", "#7f7f7f"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat_axes = axes.flatten()

    for idx, (title, key, scale, ylim) in enumerate(kpis):
        ax = flat_axes[idx]
        means = []
        stds = []
        for strat in strategies:
            strat_rows = [r for r in evaluation_rows if r["strategy"] == strat]
            mean_v, std_v, _, _, _ = _stats([r.get(key) for r in strat_rows], scale=scale)
            means.append(mean_v if mean_v is not None else 0.0)
            stds.append(std_v if std_v is not None else 0.0)

        bars = ax.bar(strategies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, mean_val in zip(bars, means):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            offset = 1.0 if y_pos >= 0 else -1.0
            ax.annotate(
                f"{mean_val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                xytext=(0, offset * 3),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle("So sánh hiệu năng điều phối: Q-learning vs Auto-dispatch (10 Eval Seeds)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_training_report(
    path: Path,
    args: argparse.Namespace,
    elapsed_training: float,
    elapsed_eval: float,
    gate_checks: Dict[str, Tuple[str, str]],
    evaluation_rows: List[Dict[str, object]],
    agent: QLearningAgent,
    best_episode: int,
    best_val_score: Optional[Tuple[float, float]],
) -> None:
    """Ghi báo cáo toàn diện training_report.txt theo chuẩn RL Standard v1.0."""
    lines = []
    lines.append("=" * 80)
    lines.append("BÁO CÁO KẾT QUẢ HUẤN LUYỆN & ĐÁNH GIÁ TABULAR Q-LEARNING")
    lines.append("Dự án: Ambulance Dispatch Simulation")
    lines.append(f"Thời gian tạo báo cáo: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. CẤU HÌNH HUẤN LUYỆN (HYPERPARAMETERS)")
    lines.append("-" * 80)
    lines.append(f"  Số episodes huấn luyện: {args.episodes}")
    lines.append(f"  Tốc độ học (alpha): {args.alpha} (adaptive_alpha=True, alpha_min={args.alpha_min}, alpha_decay={args.alpha_decay})")
    lines.append(f"  Hệ số chiết khấu (gamma): {args.gamma}")
    lines.append(f"  Khám phá (epsilon): start={args.epsilon_start}, min={args.epsilon_min}")
    lines.append(f"  Hạt giống cơ sở (seed): {args.seed}")
    lines.append(f"  Hạt giống huấn luyện: {args.seed} .. {args.seed + args.episodes - 1}")
    lines.append(f"  Hạt giống validation: {args.seed + 50_000} .. {args.seed + 50_009} (chu kỳ val: {args.val_interval} ep)")
    lines.append(f"  Hạt giống đánh giá: {args.seed + args.eval_seed_offset} .. {args.seed + args.eval_seed_offset + args.evaluation_episodes - 1}")
    lines.append(f"  Top-K điều phối mỗi slot: {args.top_k}")
    lines.append(f"  Replay buffer: capacity={agent.replay_capacity}, batch_size={agent.replay_batch_size}")
    lines.append(f"  Thời gian huấn luyện: {elapsed_training / 60:.2f} phút ({elapsed_training:.1f} giây)")
    lines.append(f"  Thời gian đánh giá: {elapsed_eval / 60:.2f} phút ({elapsed_eval:.1f} giây)")
    lines.append(f"  Checkpoint tốt nhất: Episode {best_episode} (Val score={100 * best_val_score[0]:.2f}%, {best_val_score[1]:.2f})" if best_val_score else "")
    lines.append(f"  Kích thước Q-table: {len(agent.q_table):,} cặp trạng thái-hành động")
    lines.append("")
    lines.append("2. KẾT QUẢ KIỂM CHỨNG CORRECTNESS GATE (G1 - G7)")
    lines.append("-" * 80)
    for check_name, (status, detail) in gate_checks.items():
        lines.append(f"  [{status:<4}] {check_name:<38} | {detail}")
    lines.append("")
    lines.append("3. BẢNG KẾT QUẢ CHI TIẾT TỪNG SEED ĐÁNH GIÁ (EVALUATION PER SEED)")
    lines.append("-" * 80)
    lines.append(f"  {'Seed':<8} {'Chiến lược':<15} {'Hoàn thành':<14} {'Thất bại':<12} {'Tỉ lệ (%)':<12} {'Reward':<12} {'No-op (%)':<10}")
    for r in evaluation_rows:
        succ_str = f"{100 * float(r['success_rate']):5.1f}%" if r["success_rate"] is not None else "  N/A "
        noop_str = f"{100 * float(r['decision_noop_rate']):5.1f}%" if r["decision_noop_rate"] is not None else "  N/A "
        lines.append(
            f"  {r['seed']:<8} {r['strategy']:<15} {r['served']:>3}/{r['calls']:<3} ca ({succ_str}) "
            f"{r['failed']:>3} ca      {succ_str:<12} {float(r['total_reward']):>10.2f}  {noop_str:<10}"
        )
    lines.append("")
    lines.append("4. BẢNG TỔNG KẾT THỐNG KÊ TOÀN DIỆN (MEAN ± STD, MIN, MAX)")
    lines.append("-" * 80)
    for strategy in ("Q-learning", "Auto-dispatch"):
        strat_rows = [r for r in evaluation_rows if r["strategy"] == strategy]
        lines.append(f"  [{strategy}] (N={len(strat_rows)} seeds):")
        metrics = [
            ("Served Rate (%)", "success_rate", 100.0),
            ("Failed Rate (%)", "failure_rate", 100.0),
            ("Unresolved Rate (%)", "unresolved_rate", 100.0),
            ("High-Sev Served (%)", "high_sev_served_rate", 100.0),
            ("High-Sev Resp (p)", "high_sev_response_time", 1.0),
            ("Avg Wait (p)", "avg_wait_time", 1.0),
            ("Avg Slack (p)", "avg_slack_at_dispatch", 1.0),
            ("Prio Violation (%)", "priority_violation_rate", 100.0),
            ("Episode Reward", "total_reward", 1.0),
            ("Total Distance (km)", "total_distance", 1.0),
            ("Utilization (%)", "ambulance_utilization", 100.0),
            ("No-op QĐ (%)", "decision_noop_rate", 100.0),
            ("State Hit (%)", "state_hit_rate", 100.0),
        ]
        for label, key, scale in metrics:
            mean_v, std_v, min_v, max_v, n_valid = _stats([r.get(key) for r in strat_rows], scale=scale)
            if mean_v is None:
                lines.append(f"    {label:<21}:      N/A ±    N/A  (min=     N/A, max=     N/A, n=0/{len(strat_rows)})")
            else:
                lines.append(f"    {label:<21}: {mean_v:8.2f} ± {std_v:6.2f}  (min={min_v:8.2f}, max={max_v:8.2f}, n={n_valid}/{len(strat_rows)})")
    lines.append("")
    lines.append("5. PHÂN TÍCH NGUYÊN NHÂN VÀ GIẢI PHÁP WAIT LOOP")
    lines.append("-" * 80)
    lines.append("  Nguyên nhân cốt lõi của WAIT loop:")
    lines.append("    1. Bất đối xứng Semi-MDP: Hành động WAIT chỉ kéo dài 1 phút (chiết khấu gamma^1 = 0.96),")
    lines.append("       trong khi DISPATCH kéo dài 5-15 phút đến khi có cuộc gọi mới (chiết khấu gamma^15 = 0.54).")
    lines.append("    2. Độ trễ phần thưởng hoàn thành: Phần thưởng hoàn thành (+50, +65, +70) chỉ xuất hiện")
    lines.append("       sau 20-30 phút khi xe đưa bệnh nhân đến bệnh viện, trong khi transition của DISPATCH")
    lines.append("       đã bị đóng từ phút thứ 5-10 mà không nhận được tín chỉ hoàn thành nào.")
    lines.append("    3. Gán nhầm phần thưởng: Khi xe hoàn thành ở phút 30, nếu agent đang chọn WAIT cho một")
    lines.append("       cuộc gọi khác, phần thưởng +70 lại được cộng vào macro_reward của hành động WAIT!")
    lines.append("    4. Tie-breaking mặc định chọn 0: Trên các trạng thái chưa gặp (unseen state), giá trị Q")
    lines.append("       đều bằng 0.0, thuật toán tie-break chọn phần tử đầu tiên (action 0 = WAIT).")
    lines.append("")
    lines.append("  Giải pháp khắc phục:")
    lines.append("    1. Immediate Dispatch Credit Assignment: Gán tín chỉ hoàn thành kỳ vọng trực tiếp vào")
    lines.append("       thời điểm điều phối cho các cuộc gọi khả thi (ETA <= deadline - t), đồng thời loại trừ")
    lines.append("       completion_bonus xuất hiện trễ ở các bước sau để tránh gán nhầm cho WAIT.")
    lines.append("    2. Smart Deterministic Tie-breaking: Khi các giá trị Q bằng nhau (trạng thái chưa gặp),")
    lines.append("       ưu tiên hành động điều phối tốt nhất thay vì mặc định chọn WAIT, loại bỏ hoàn toàn")
    lines.append("       hiện tượng kẹt trong vòng lặp chờ vô tận trên các trạng thái mới.")
    lines.append("=" * 80)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Đọc tùy chọn, huấn luyện, kiểm tra Correctness Gate, đánh giá và ghi kết quả."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="Số episode huấn luyện.")
    parser.add_argument("--evaluation-episodes", type=int, default=10, help="Số seed đánh giá (Mục 5.3: 10).")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--max-overtime", type=int, default=DEFAULT_MAX_OVERTIME)
    parser.add_argument("--poisson", type=float, default=DEFAULT_POISSON_LAMBDA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--alpha-min", type=float, default=0.015)
    parser.add_argument("--alpha-decay", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--initial-q", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=40)
    parser.add_argument("--eval-seed-offset", type=int, default=200_000, help="Offset cho evaluation seeds (mặc định 200_000 -> seeds 200042..200051).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/q_learning_final_v2"))
    args = parser.parse_args()

    if args.episodes < 1 or args.evaluation_episodes < 1:
        parser.error("episodes và evaluation-episodes phải lớn hơn 0.")
    if args.max_time < 1 or args.max_overtime < 0:
        parser.error("max-time phải lớn hơn 0 và max-overtime không được âm.")
    if not math.isfinite(args.poisson) or args.poisson < 0:
        parser.error("poisson phải là số hữu hạn và không âm.")

    def make_env(seed: int) -> AmbulanceDispatchEnv:
        return AmbulanceDispatchEnv(
            num_ambulances=DEFAULT_NUM_AMBULANCES,
            max_time=args.max_time,
            max_overtime=args.max_overtime,
            poisson_lambda=args.poisson,
            seed=seed,
        )

    decay_steps = max(1, int(0.8 * args.episodes))
    epsilon_decay = (args.epsilon_min / max(args.epsilon_min, args.epsilon_start)) ** (1.0 / decay_steps)

    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=args.seed,
        adaptive_alpha=True,
        alpha_min=args.alpha_min,
        alpha_decay=args.alpha_decay,
        initial_q=args.initial_q,
        replay_capacity=15000,
        replay_batch_size=128,
    )
    adapter = AmbulanceDispatchAdapter(top_k_per_slot=args.top_k)

    # Tập validation độc lập (seed + 50_000) tách biệt khỏi train (seed + 0..) và test (seed + 100_000..).
    val_seeds = [args.seed + 50_000 + i for i in range(10)]
    best_val_score: Optional[Tuple[float, float]] = None
    best_q_table: Optional[Dict[Tuple[str, int], float]] = None
    best_episode = 0

    training_rows: List[Dict[str, object]] = []
    training_started_at = time.perf_counter()
    for episode in range(args.episodes):
        seed = args.seed + episode
        result = run_episode(
            make_env(seed), adapter, agent, training=True, seed=seed
        )
        result["episode"] = episode + 1
        result["epsilon"] = agent.epsilon
        result["q_table_size"] = len(agent.q_table)
        training_rows.append(result)

        val_note = ""
        if args.val_interval > 0 and ((episode + 1) % args.val_interval == 0 or episode + 1 == args.episodes):
            val_succ, val_rew = evaluate_policy(make_env, adapter, agent, val_seeds)
            score = (round(val_succ, 4), val_rew)
            if best_val_score is None or score > best_val_score:
                best_val_score = score
                best_q_table = dict(agent.q_table)
                best_episode = episode + 1
            val_note = f" | Val: {100 * val_succ:.1f}% ({val_rew:.1f})"

        print_episode_progress(
            phase="Huấn luyện",
            episode=episode + 1,
            total_episodes=args.episodes,
            started_at=training_started_at,
            result=result,
            details=(
                f"eps={agent.epsilon:.3f} | Q={len(agent.q_table)} cặp{val_note}"
            ),
        )
        agent.decay_exploration()

    if best_q_table is not None:
        agent.q_table = best_q_table
        print(
            f"[Checkpoint] Đã chọn Q-table tốt nhất từ episode {best_episode} "
            f"(Val score={100 * best_val_score[0]:.2f}%, {best_val_score[1]:.2f}).",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_metrics.csv", training_rows)
    agent.save(args.output_dir / "q_table.json")

    q_table_before_eval = dict(agent.q_table)
    visits_before_eval = dict(agent.visit_counts)

    evaluation_rows: List[Dict[str, object]] = []
    sample_log = ""
    evaluation_started_at = time.perf_counter()
    for index in range(args.evaluation_episodes):
        seed = args.seed + args.eval_seed_offset + index
        q_result = run_episode(
            make_env(seed),
            adapter,
            agent,
            training=False,
            seed=seed,
            capture_log=index == 0,
        )
        if index == 0:
            sample_log = str(q_result["log"])
        q_result.update({"episode": index + 1, "strategy": "Q-learning"})
        evaluation_rows.append(q_result)

        baseline = run_episode(
            make_env(seed),
            adapter,
            agent=None,
            training=False,
            seed=seed,
        )
        baseline.update({"episode": index + 1, "strategy": "Auto-dispatch"})
        evaluation_rows.append(baseline)
        base_succ = baseline["success_rate"]
        base_succ_str = f"{100 * float(base_succ):.1f}%" if base_succ is not None else "N/A"
        print_episode_progress(
            phase="Đánh giá",
            episode=index + 1,
            total_episodes=args.evaluation_episodes,
            started_at=evaluation_started_at,
            result=q_result,
            details=(
                f"Auto-dispatch={base_succ_str} "
                f"({float(baseline['total_reward']):.1f})"
            ),
        )

    gate_checks = verify_correctness_gate(
        make_env=make_env,
        adapter=adapter,
        agent=agent,
        evaluation_rows=evaluation_rows,
        q_table_before_eval=q_table_before_eval,
        visits_before_eval=visits_before_eval,
        repro_seed=args.seed + args.eval_seed_offset,
    )

    write_csv(args.output_dir / "evaluation_per_seed.csv", evaluation_rows)
    write_csv(args.output_dir / "evaluation_metrics.csv", evaluation_rows)
    write_summary_csv(args.output_dir / "evaluation_summary.csv", evaluation_rows)
    (args.output_dir / "q_learning_episode.log").write_text(sample_log + "\n", encoding="utf-8")
    plot_results(training_rows, evaluation_rows, args.output_dir / "q_learning_results.png")
    plot_evaluation_comparison(evaluation_rows, args.output_dir / "evaluation_comparison.png")

    elapsed_training = evaluation_started_at - training_started_at
    elapsed_eval = time.perf_counter() - evaluation_started_at

    write_training_report(
        path=args.output_dir / "training_report.txt",
        args=args,
        elapsed_training=elapsed_training,
        elapsed_eval=elapsed_eval,
        gate_checks=gate_checks,
        evaluation_rows=evaluation_rows,
        agent=agent,
        best_episode=best_episode,
        best_val_score=best_val_score,
    )

    total_visits = sum(agent.visit_counts.values())
    avg_visits = total_visits / max(1, len(agent.visit_counts))

    print("\n" + "=" * 95)
    print("CORRECTNESS GATE (RL Standard v1.0 - Mục 7)")
    print("-" * 95)
    for check_name, (status, detail) in gate_checks.items():
        print(f"  {check_name:<38}: {status:<10} | {detail}")

    print("\n" + "=" * 95)
    print(f"TỔNG KẾT ĐÁNH GIÁ (High-Severity = Mức {HIGH_SEVERITY_LEVEL}, bao gồm penalty t=0)")
    print("-" * 95)
    for strategy in ("Q-learning", "Auto-dispatch"):
        rows = [row for row in evaluation_rows if row["strategy"] == strategy]
        print_final_summary(strategy, rows)
    print("-" * 95)
    print(
        f"Checkpoint: episode {best_episode} | Q-table: {len(agent.q_table):,} cặp "
        f"(lượt thăm qua {args.episodes} ep train: {total_visits:,}, TB {avg_visits:.1f} lượt/cặp)"
    )
    print(f"Thư mục kết quả: {args.output_dir.resolve()}")
    print("=" * 95)


if __name__ == "__main__":
    main()
