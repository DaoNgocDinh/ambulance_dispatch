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
"""

from __future__ import annotations

from dispatch_env.environment import AmbulanceDispatchEnv
from dispatch_env.types import AmbulanceStatus


class GroupEnvAdapter:
    """Bo chuyen doi env cua nhom -> (state tuple, action int, reward, done)."""

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
    def discretize_state(self) -> tuple:
        """Bien obs phuc tap cua env thanh tuple lam key cho Q-table.

        state = ( (xe1_ranh?, ..., xe4_ranh?),
                  ((sev_ca1, wait_bucket_ca1), ..., (sev_ca3, wait_bucket_ca3)) )
        Ca trong slot duoc dien (0, 0). Kich thuoc toi da:
        2^4 x (3 muc nghiem trong x 3 bucket + 1 o trong)^3 = 16.000 state.
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
        return (amb_part, tuple(call_part))

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

        Reward do env tu tinh theo dung thiet ke cua nhom:
        -1.0 x muc do / phut cho moi ca dang cho;
        -50 x muc do cho ca that bai / bi tu choi;
        +100 x muc do cho ca hoan thanh (+50 x muc do neu hoan hao).
        """
        pair = self._decode(action_idx)
        # env.step chap nhan tuple (ma_xe, ma_ca) hoac None (cho).
        _obs, reward, done, _info = self.env.step(pair)
        return self.discretize_state(), float(reward), bool(done)

    def get_summary(self) -> dict:
        """Chi so tong ket episode (phuc vu ghi log)."""
        return {
            "minute": self.env.current_minute,
            "served": self.env.served_calls_count,
            "failed": self.env.failed_calls_count,
            "perfect": self.env.perfect_calls_count,
        }
