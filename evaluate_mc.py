"""Danh gia chinh sach theo QUY CHUAN RL cua nhom (RL_ALGORITHM_STANDARD v1.0).

Danh gia MOT checkpoint (hoac Heuristic baseline) tren 10 Evaluation Seed
chung, tinh KPI theo Muc 6 va tong hop Mean/Std/Min/Max theo Muc 9:

- Che do Evaluation thuan: policy tham lam (training=False), KHONG cap
    nhat Q, KHONG tham do ngau nhien (Muc 5.2).
- KPI: Served/Failed Rate (mau so = tong so ca phat sinh), KPI theo
    tung muc nghiem trong, High-Severity (bao cao ca 2 dinh nghia {3} va
    {2,3} vi V-03 chua chot), Response Time = arrived - created,
    Deadline Success Rate, Average Waiting Time, Episode Reward +
    Reward Components tu info["reward_breakdown"], Total Distance,
    Ambulance Utilization (dinh nghia ben duoi), Unresolved khi
    Truncation (chi gom pending + waiting + at_scene, Muc 6.2.2).
- Episode Reward cong ca khoan penalty t=0 cua reset() (doc tu
    env.total_reward/last_reward_breakdown ngay sau khi tao env) va
    DOI SOAT voi env.total_reward cuoi episode (Muc 10 / V-06).
- Correctness (Muc 7): dem vi pham Action Mask (phai = 0) va kiem tra
    Q-table KHONG doi sau danh gia.
- Chua tinh: Average Slack Time at Dispatch va Priority Violation
    (V-04: nhom chua co dinh nghia chung) - khai bao ro trong JSON.

Ambulance Utilization (dinh nghia cua evaluator nay, ap dung nhu nhau
cho moi policy): tong so phut-xe khong o trang thai IDLE / (so xe x
tong so phut episode).

"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from map_env import CityMap
from dispatch_env import (
    AmbulanceDispatchEnv,
    AmbulanceStatus,
    CallStatus,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_MAX_TIME,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_MAX_PENDING_CALLS,
)

from group_adapter import GroupEnvAdapter
from mc_agent import MonteCarloAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_EVAL_SEEDS = list(range(1000, 1010))
UNRESOLVED_STATUSES = (
    CallStatus.PENDING,
    CallStatus.WAITING_FOR_AMBULANCE,
    CallStatus.AT_SCENE,
)
REWARD_COMPONENTS = (
    "wait_penalty",
    "failure_penalty",
    "distance_penalty",
    "completion_bonus",
)


def build_env(seed: int, cfg: dict) -> AmbulanceDispatchEnv:
    """Tao env moi theo DUNG cau hinh chuan (nhu run_simulation.py)."""
    city_map = CityMap(seed=seed)
    return AmbulanceDispatchEnv(
        city_map=city_map,
        num_ambulances=cfg["num_ambulances"],
        max_time=cfg["max_time"],
        max_overtime=cfg["max_overtime"],
        poisson_lambda=cfg["poisson_lambda"],
        max_pending_calls=cfg["max_pending_calls"],
        seed=seed,
    )


def run_episode(policy: str, env: AmbulanceDispatchEnv,
                adapter: GroupEnvAdapter | None,
                agent: MonteCarloAgent | None) -> dict:
    """Chay 1 episode danh gia va tinh KPI theo Muc 6."""
    # Khoan t=0 cua reset(): env da cong vao total_reward nhung khong
    # tra qua step() nao -> cong thu cong, khai bao minh bach (V-06).
    t0_reward = float(env.total_reward)
    components = {k: float(env.last_reward_breakdown.get(k, 0.0))
                for k in REWARD_COMPONENTS}

    step_reward_sum = 0.0
    busy_minutes = 0
    steps = 0
    mask_violations = 0
    truncated = False
    info: dict = {}

    state = adapter.discretize_state() if adapter is not None else None
    done = False
    while not done:
        if policy == "mc":
            valid = adapter.get_valid_actions()
            action_idx = agent.select_action(state, valid, training=False)
            # Kiem tra mask: cap duoc dieu phai la xe RANH + ca PENDING.
            pair = adapter._decode(action_idx)
            if pair is not None:
                aid, call_id = pair
                amb_ok = (env.ambulances[aid].status
                        == AmbulanceStatus.IDLE)
                call_ok = any(c.call_id == call_id
                            for c in env.pending_calls)
                if not (amb_ok and call_ok):
                    mask_violations += 1
            state, reward, done = adapter.step(action_idx)
            info = adapter.last_info
        else:  # heuristic baseline cua env
            _obs, reward, done, info = env.step(auto_dispatch=True)

        step_reward_sum += float(reward)
        steps += 1
        breakdown = info.get("reward_breakdown", {})
        for k in REWARD_COMPONENTS:
            components[k] += float(breakdown.get(k, 0.0))
        busy_minutes += sum(
            1 for a in env.ambulances.values()
            if a.status != AmbulanceStatus.IDLE
        )
        truncated = bool(info.get("truncated", False))

    # ---- Tong hop tu env.calls (chi doc, Muc 6.2.5) ----
    calls = list(env.calls.values())
    total_generated = len(calls)
    served = env.served_calls_count
    failed = env.failed_calls_count
    if truncated:
        backlog = env.get_backlog()
        unresolved = (len(backlog["pending_calls"])
                    + len(backlog["waiting_calls"])
                    + len(backlog["at_scene_calls"]))
    else:
        unresolved = sum(1 for c in calls if c.status in UNRESOLVED_STATUSES)
    count_diff = total_generated - (served + failed + unresolved)

    def group_stats(levels: set[int]) -> dict:
        grp = [c for c in calls if c.severity in levels]
        g_served = [c for c in grp if c.status == CallStatus.COMPLETED]
        g_failed = [c for c in grp if c.status == CallStatus.FAILED]
        resolved = len(g_served) + len(g_failed)
        responses = [
            c.arrived_at_scene_time - c.created_time
            for c in g_served if c.arrived_at_scene_time is not None
        ]
        on_time = sum(
            1 for c in grp
            if c.arrived_at_scene_time is not None
            and c.arrived_at_scene_time <= c.deadline
        )
        return {
            "generated": len(grp),
            "served_rate": (len(g_served) / len(grp)) if grp else 0.0,
            "failed_rate": (len(g_failed) / len(grp)) if grp else 0.0,
            "mean_response_time": (statistics.fmean(responses)
                                if responses else 0.0),
            "deadline_success_rate": (on_time / resolved) if resolved else 0.0,
        }

    resolved_calls = [c for c in calls
                    if c.status in (CallStatus.COMPLETED,
                                    CallStatus.FAILED)]
    avg_wait = (statistics.fmean([c.wait_time for c in resolved_calls])
                if resolved_calls else 0.0)

    episode_reward = t0_reward + step_reward_sum
    row = {
        "total_generated": total_generated,
        "served": served,
        "failed": failed,
        "perfect": env.perfect_calls_count,
        "unresolved": unresolved,
        "count_check_diff": count_diff,
        "truncated": int(truncated),
        "episode_minutes": env.current_minute,
        "served_rate": served / total_generated if total_generated else 0.0,
        "failed_rate": failed / total_generated if total_generated else 0.0,
        "unresolved_rate": (unresolved / total_generated
                            if total_generated else 0.0),
        "avg_waiting_time": avg_wait,
        "episode_reward": round(episode_reward, 2),
        "reward_crosscheck_diff": round(episode_reward
                                        - float(env.total_reward), 4),
        "total_distance_km": round(
            sum(a.total_distance_traveled
                for a in env.ambulances.values()), 2),
        "utilization": (busy_minutes / (len(env.ambulances) * steps)
                        if steps else 0.0),
        "mask_violations": mask_violations,
    }
    for k in REWARD_COMPONENTS:
        row[f"reward_{k}"] = round(components[k], 2)
    for sev in (1, 2, 3):
        s = group_stats({sev})
        row[f"sev{sev}_generated"] = s["generated"]
        row[f"sev{sev}_served_rate"] = round(s["served_rate"], 4)
        row[f"sev{sev}_failed_rate"] = round(s["failed_rate"], 4)
    for tag, levels in (("sev3", {3}), ("sev23", {2, 3})):
        s = group_stats(levels)
        row[f"{tag}_served_rate"] = round(s["served_rate"], 4)
        row[f"{tag}_failed_rate"] = round(s["failed_rate"], 4)
        row[f"{tag}_response_time"] = round(s["mean_response_time"], 2)
        row[f"{tag}_deadline_success_rate"] = round(
            s["deadline_success_rate"], 4)
    return row


def aggregate(rows: list[dict]) -> dict:
    """Mean/Std/Min/Max cho tung KPI so (Muc 9). Std = do lech mau."""
    agg = {}
    for key in rows[0]:
        if key == "seed":
            continue
        values = [float(r[key]) for r in rows]
        agg[key] = {
            "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1
                else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return agg


def q_snapshot(agent: MonteCarloAgent) -> tuple:
    """Dau van tay cua Q-table de chung minh evaluation khong hoc."""
    return (
        len(agent.q_table),
        round(sum(float(v.sum()) for v in agent.q_table.values()), 6),
        round(sum(float(v.sum()) for v in agent.returns_count.values()), 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá MC/Heuristic theo quy chuẩn RL của nhóm")
    parser.add_argument("--policy", choices=["mc", "heuristic"],
                        default="mc")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint MC (bắt buộc khi --policy mc)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Danh sách seed cách nhau dấu phẩy "
                            "(mặc định 1000..1009)")
    parser.add_argument("--ambulances", type=int,
                        default=DEFAULT_NUM_AMBULANCES)
    parser.add_argument("--poisson", type=float,
                        default=DEFAULT_POISSON_LAMBDA)
    parser.add_argument("--max_time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--max_overtime", type=int,
                        default=DEFAULT_MAX_OVERTIME)
    parser.add_argument("--max_pending", type=int,
                        default=DEFAULT_MAX_PENDING_CALLS)
    parser.add_argument("--out-prefix", type=str, default=None)
    args = parser.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")]
            if args.seeds else DEFAULT_EVAL_SEEDS)
    cfg = {
        "num_ambulances": args.ambulances,
        "max_time": args.max_time,
        "max_overtime": args.max_overtime,
        "poisson_lambda": args.poisson,
        "max_pending_calls": args.max_pending,
    }
    prefix = args.out_prefix or f"evaluation_{args.policy}"

    agent = None
    before = None
    if args.policy == "mc":
        if not args.model:
            parser.error("--policy mc bat buoc co --model <checkpoint.pkl>")
        agent = MonteCarloAgent.load(args.model)
        before = q_snapshot(agent)

    print(f"[*] Policy: {args.policy} | Seeds: {seeds} | Config: {cfg}")
    rows: list[dict] = []
    for seed in seeds:
        env = build_env(seed, cfg)
        adapter = None
        if args.policy == "mc":
            adapter = GroupEnvAdapter(env=env)
            if agent.number_of_actions != adapter.number_of_actions:
                raise ValueError(
                    f"Model co {agent.number_of_actions} action, cau hinh "
                    f"danh gia can {adapter.number_of_actions} action "
                    f"({cfg['num_ambulances']} xe). Train lai dung cau hinh."
                )
        row = run_episode(args.policy, env, adapter, agent)
        row = {"seed": seed, **row}
        rows.append(row)
        print(f"    seed {seed}: served {row['served']}/"
            f"{row['total_generated']} failed {row['failed']} "
            f"unresolved {row['unresolved']} | reward "
            f"{row['episode_reward']:.2f} | mask_violations "
            f"{row['mask_violations']} | count_diff "
            f"{row['count_check_diff']} | reward_diff "
            f"{row['reward_crosscheck_diff']}")

    policy_unchanged = None
    if args.policy == "mc":
        policy_unchanged = q_snapshot(agent) == before
        print(f"[*] Q-table khong doi sau danh gia: {policy_unchanged}")

    agg = aggregate(rows)

    csv_path = Path(f"{prefix}_per_seed.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy": args.policy,
        "checkpoint": args.model,
        "evaluation_seeds": seeds,
        "env_config": cfg,
        "kpi_definitions": {
            "served_rate": "COMPLETED / total_generated",
            "failed_rate": "FAILED (gom ca queue_overflow) / total_generated",
            "unresolved": "PENDING + WAITING_FOR_AMBULANCE + AT_SCENE "
                        "tai truncation (Muc 6.2.2)",
            "response_time": "arrived_at_scene_time - created_time "
                            "(chi tinh tren ca COMPLETED)",
            "deadline_success_rate": "ca toi hien truong <= deadline / "
                                    "ca da resolved (COMPLETED+FAILED)",
            "avg_waiting_time": "trung binh wait_time tren ca resolved",
            "episode_reward": "reward t=0 cua reset + tong reward step(); "
                            "doi soat env.total_reward",
            "utilization": "phut-xe khong IDLE / (so xe x so phut episode)",
            "high_severity": "bao cao ca {3} (sev3) va {2,3} (sev23); "
                            "V-03 chua chot dinh nghia chung",
        },
        "not_computed": [
            "average_slack_time_at_dispatch (V-04: chua co dinh nghia "
            "chung cua nhom)",
            "priority_violation (V-04: chua co dinh nghia chung cua nhom)",
        ],
        "correctness": {
            "total_mask_violations": sum(r["mask_violations"]
                                        for r in rows),
            "policy_unchanged_after_evaluation": policy_unchanged,
            "max_abs_count_check_diff": max(
                abs(r["count_check_diff"]) for r in rows),
            "max_abs_reward_crosscheck_diff": max(
                abs(r["reward_crosscheck_diff"]) for r in rows),
        },
        "aggregate": agg,
        "per_seed": rows,
    }
    json_path = Path(f"{prefix}_summary.json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n===== TỔNG HỢP (mean ± std | min | max) =====")
    for key in ("served_rate", "failed_rate", "unresolved_rate",
                "avg_waiting_time", "episode_reward",
                "sev3_served_rate", "sev3_failed_rate",
                "sev3_response_time", "sev3_deadline_success_rate",
                "sev23_served_rate", "total_distance_km", "utilization"):
        a = agg[key]
        print(f"  {key:<28} {a['mean']:>10.4f} ± {a['std']:<8.4f} "
            f"| {a['min']:>10.4f} | {a['max']:>10.4f}")
    print(f"\n[+] Đã ghi: {csv_path} và {json_path}")


if __name__ == "__main__":
    main()
