"""Adapter: moi truong dieu phoi cua nhom (DaoNgocDinh/ambulance_dispatch)
-> interface chuan cho tabular on-policy MC Control.

Tai sao can adapter? Env cua nhom co 2 dac diem khac env tham khao:
1. Hanh dong la CAP (ma_xe, ma_cuoc_goi), khong phai so nguyen 0..3,
    va KHONG co action "cho" rieng (cho = khong truyen action).
2. State la dict long nhau (80 nut, vi tri lien tuc) -> phai roi rac
    hoa moi lam key cho Q-table duoc.

Adapter giai quyet ca 2 viec:
- Ma hoa moi cap (xe, ca) thanh so nguyen 0..12 (12 cap + 1 WAIT),
    voi action masking: chi cap (xe RANH, ca PENDING) moi hop le.
- Rui ro quan trong: "slot" cua ca goi chi co nghia khi danh sach ca
    duoc sap xep DETERMINISTIC giong nhau o moi buoc -> _sorted_pending()
    sap theo (-severity, -wait_time, call_id), trung voi tieu chi uu tien
    cua che do auto_dispatch trong env.

PHIEN BAN STATE v2 (STATE_VERSION = 2):
State duoc bo sung NHOM 3 - thong tin KHOANG CACH toi ca o dau hang
cho (slot 0 cua _sorted_pending): (ma xe ranh co ETA ngan nhat, bucket
ETA). ETA tinh bang DUNG ham city_map.route_from_road_to_road ma che
do auto_dispatch cua env dung de chon xe, nen agent va heuristic gio
"nhin" cung mot dai luong khoang cach (du agent chi thay dang tho).
Q-table train bang state v1 (khong co nhom 3) KHONG dung lai duoc:
khoa state khac han, phai train lai tu dau.
"""

from __future__ import annotations

from dispatch_env.environment import AmbulanceDispatchEnv
from dispatch_env.types import AmbulanceStatus


class GroupEnvAdapter:
    """Bo chuyen doi env cua nhom -> (state tuple, action int, reward, done)."""

    STATE_VERSION = 2  # v2: state co them nhom khoang cach (nhom 3)

    def __init__(
        self,
        env: AmbulanceDispatchEnv | None = None,
        max_time: int = 360,
        poisson_lambda: float = 0.5,
        max_pending_calls: int = 3,
        num_ambulances: int = 4,
        seed: int = 42,
    ) -> None:
        # Neu khong truyen env san thi tu tao theo tham so.
        # LUU Y: poisson_lambda mac dinh trong code env la 0.05, nhung
        # README va run_simulation.py cua nhom dung 0.5 -> o day lay 0.5.
        if env is None:
            env = AmbulanceDispatchEnv(
                max_time=max_time,
                poisson_lambda=poisson_lambda,
                max_pending_calls=max_pending_calls,
                num_ambulances=num_ambulances,
                seed=seed,
            )
        self.env = env
        self.num_ambulances = env.num_ambulances
        self.max_pending = env.max_pending_calls
        # So cap (xe, ca) toi da = 4 xe x 3 ca = 12; index cuoi la WAIT.
        self.num_pair_actions = self.num_ambulances * self.max_pending
        self.WAIT = self.num_pair_actions
        # Tong so hanh dong agent thay: 13 (0..11 dieu xe, 12 cho).
        self.number_of_actions = self.num_pair_actions + 1
        # info cua step() gan nhat (reward_breakdown, truncated, backlog...)
        # de evaluator ben ngoai doc theo quy chuan - adapter khong dung toi.
        self.last_info: dict = {}

    # ------------------------------------------------------------------
    # 1. Sap xep ca PENDING theo thu tu deterministic (QUAN TRONG).
    # ------------------------------------------------------------------
    def _sorted_pending(self):
        """Danh sach ca PENDING, sap xep on dinh de "slot" co nghia.

        Thu tu: nghiem trong giam dan -> cho lau giam dan -> ma ca tang.
        Trung voi tieu chi cua auto_dispatch trong env, nen state va
        action luon "hieu" slot theo cung mot cach.
        """
        return sorted(
            self.env.pending_calls,
            key=lambda c: (-c.severity, -c.wait_time, c.call_id),
        )

    @staticmethod
    def _wait_bucket(wait_time: float) -> int:
        """Roi rac hoa thoi gian cho (phut) thanh 3 nhom."""
        if wait_time < 5:
            return 0
        if wait_time < 15:
            return 1
        return 2

    # ------------------------------------------------------------------
    # 2. Roi rac hoa state: (trang thai 4 xe, thong tin 3 ca cho).
    # ------------------------------------------------------------------
    # Nguong chia bucket ETA cho ca dau hang cho (phut): <8 gan,
    # 8..18 vua, >18 xa. Chon theo deadline muc 3 (30 phut): ETA > 18
    # cong thoi gian cho da tich luy thuong khien ca muc 3 rat kho kip.
    ETA_BUCKET_EDGES = (8.0, 18.0)

    def _top_call_distance_features(self, pending) -> tuple:
        """Dac trung khoang cach cho ca o slot 0 (ca quan trong nhat).

        Tra ve (nearest_amb_id, eta_bucket):
        - nearest_amb_id: ma xe RANH co ETA toi ca dau hang ngan
            nhat (0 neu khong co ca cho hoac khong co xe ranh).
        - eta_bucket: 0 = khong ap dung, 1 = ETA < 8 phut,
            2 = 8..18 phut, 3 = ETA > 18 phut.
        ETA tinh bang route_from_road_to_road(by_travel_time=True) -
        dung ham env dung trong auto_dispatch (dispatch_env), chi DOC
        trang thai env, khong thay doi gi.
        """
        idle = [
            self.env.ambulances[aid]
            for aid in range(1, self.num_ambulances + 1)
            if self.env.ambulances[aid].status == AmbulanceStatus.IDLE
        ]
        if not pending or not idle:
            return (0, 0)
        top = pending[0]
        best_id, best_eta = 0, float("inf")
        for amb in idle:
            r = self.env.city_map.route_from_road_to_road(
                start_edge=amb.current_edge,
                start_dist=amb.distance_from_start,
                dest_edge=top.edge,
                dest_dist=top.distance_from_start,
                by_travel_time=True,
            )
            eta = float(r["total_time_minutes"])
            if eta < best_eta:
                best_eta, best_id = eta, amb.ambulance_id
        lo, hi = self.ETA_BUCKET_EDGES
        bucket = 1 if best_eta < lo else (2 if best_eta <= hi else 3)
        return (best_id, bucket)

    def discretize_state(self) -> tuple:
        """Bien obs phuc tap cua env thanh tuple lam key cho Q-table.

        state = ( (xe1_ranh?, ..., xeN_ranh?),
                ((sev_ca1, wait_bucket_ca1), ..., (sev_ca3, wait_bucket_ca3)),
                (nearest_amb_id, eta_bucket) )   <- nhom 3, state v2
        Ca trong slot duoc dien (0, 0). Nhom 3 mo ta khoang cach toi
        ca dau hang cho (xem _top_call_distance_features). Kich thuoc
        ly thuyet toi da o cau hinh 6 xe: 2^6 x 10^3 x (6+1) x 4
        = 1.792.000 state, nhung so state thuc te gap duoc nho hon
        nhieu vi nearest_amb_id chi co the la mot xe dang RANH va
        nhom 3 = (0, 0) moi khi khong co ca cho.
        """
        # Nhom 1: moi xe -> 1 neu RANH, 0 neu ban (dang chay nhiem vu).
        amb_part = tuple(
            1 if self.env.ambulances[aid].status == AmbulanceStatus.IDLE else 0
            for aid in range(1, self.num_ambulances + 1)
        )
        # Nhom 2: toi da 3 ca PENDING theo thu tu _sorted_pending().
        pending = self._sorted_pending()
        call_part = []
        for slot in range(self.max_pending):
            if slot < len(pending):
                call = pending[slot]
                call_part.append(
                    (call.severity, self._wait_bucket(call.wait_time))
                )
            else:
                call_part.append((0, 0))  # slot trong, khong co ca
        # Nhom 3 (v2): khoang cach toi ca dau hang cho.
        dist_part = self._top_call_distance_features(pending)
        return (amb_part, tuple(call_part), dist_part)

    def reset(self, seed: int | None = None) -> tuple:
        """Reset env, tra ve state da roi rac hoa."""
        self.env.reset(seed=seed)
        return self.discretize_state()

    # ------------------------------------------------------------------
    # 3. Ma hoa / giai ma hanh dong: int <-> (ma_xe, ma_ca).
    # ------------------------------------------------------------------
    def _decode(self, action_idx: int):
        """int -> (ma_xe, ma_ca) de truyen cho env, hoac None neu WAIT.

        Ham nay chi goi voi action_idx nam trong get_valid_actions(),
        nen luon giai ma duoc; cac nhanh phong thu ben duoi de code
        khong bao gio crash neu co lech pha giua state va action.
        """
        if action_idx == self.WAIT:
            return None  # cho = khong dieu xe
        aid = action_idx // self.max_pending + 1
        slot = action_idx % self.max_pending
        pending = self._sorted_pending()
        if slot >= len(pending):
            return None
        amb = self.env.ambulances.get(aid)
        if amb is None or amb.status != AmbulanceStatus.IDLE:
            return None
        return (aid, pending[slot].call_id)

    def get_valid_actions(self) -> list[int]:
        """Danh sach index hop le: cac cap (xe RANH, ca PENDING) + WAIT.

        Day chinh la action masking: agent khong bao gio duoc chon
        xe ban hoac ca khong ton tai.
        """
        valid: list[int] = []
        pending = self._sorted_pending()
        for aid in range(1, self.num_ambulances + 1):
            if self.env.ambulances[aid].status != AmbulanceStatus.IDLE:
                continue  # xe ban -> bo qua toan bo cap cua xe nay
            for slot in range(len(pending)):
                valid.append((aid - 1) * self.max_pending + slot)
        valid.append(self.WAIT)  # cho luon la lua chon hop le
        return valid

    def describe_action(self, action_idx: int) -> str:
        """Dien giai tieng Viet cua action (tien debug / bao cao)."""
        if action_idx == self.WAIT:
            return "WAIT: khong dieu xe trong phut nay"
        aid = action_idx // self.max_pending + 1
        slot = action_idx % self.max_pending
        pending = self._sorted_pending()
        if slot < len(pending):
            c = pending[slot]
            return (
                f"Dieu xe {aid} cho ca #{c.call_id} "
                f"(muc {c.severity}, da cho {c.wait_time:.0f} phut)"
            )
        return f"Dieu xe {aid} (slot trong)"

    # ------------------------------------------------------------------
    # 4. Thuc hien 1 buoc MDP 1 phut cua env.
    # ------------------------------------------------------------------
    def step(self, action_idx: int) -> tuple[tuple, float, bool]:
        """Nhan action dang int, tra ve (next_state, reward, done) chuan.

        Reward lay NGUYEN tu env (khong tu tinh lai), theo REWARD_CONFIG
        dang thuc thi trong dispatch_env/types.py:
        cho -0.5/-1.0/-1.5 diem/phut theo muc 1/2/3; that bai
        -30/-60/-120; hoan thanh +50/+65/+70; hoan hao +70/+80/+100;
        phat quang duong -0.05/km. Chi tiet tung buoc nam trong
        self.last_info["reward_breakdown"] sau moi lan step.
        """
        pair = self._decode(action_idx)
        # env.step chap nhan tuple (ma_xe, ma_ca) hoac None (cho).
        _obs, reward, done, _info = self.env.step(pair)
        self.last_info = _info
        return self.discretize_state(), float(reward), bool(done)

    def get_summary(self) -> dict:
        """Chi so tong ket episode (phuc vu ghi log)."""
        return {
            "minute": self.env.current_minute,
            "served": self.env.served_calls_count,
            "failed": self.env.failed_calls_count,
            "perfect": self.env.perfect_calls_count,
        }
