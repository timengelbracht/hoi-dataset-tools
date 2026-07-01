from statistics import median
from typing import List, Tuple

class TimeAligner:
    """
    Converts timestamps of *sensor* into Aria-time:
        aria_ts = sensor_ts + delta
    """

    # ---------- constructor from two QR-code pairs ----------
    def __init__(self, aria_pair: Tuple[int, int], sensor_pair: Tuple[int, int]):
        aria_dev,  aria_utc  = aria_pair
        sens_dev,  sens_utc  = sensor_pair
        aria_off   = aria_utc - aria_dev
        sens_off   = sens_utc - sens_dev
        self.delta = sens_off - aria_off         # eq. (1)

    # ---------- constructor from event pairs (no QR) ----------
    @classmethod
    def from_event_pairs(cls, event_pairs: List[Tuple[int, int]]):
        """
        event_pairs = [(sensor1_ts, sensor2_ts), …]  (same event seen by both)
        Builds an aligner that maps sensor2 → sensor1.
        """
        if not event_pairs:
            raise ValueError("Need at least one (sensor1_ts, sensor2_ts) pair")
        d = median(s1 - s2 for s1, s2 in event_pairs)   # eq. (2)
        obj = cls.__new__(cls)        # bypass __init__
        obj.delta = int(d)
        return obj

    # ---------- compose two aligners ----------
    @classmethod
    def chain(cls, aligner_a_b, aligner_b_c):
        """
        Returns an aligner that maps C → A:
            (A ← B) + (B ← C)  ⇒  (A ← C)
        """
        obj = cls.__new__(cls)
        obj.delta = aligner_a_b.delta + aligner_b_c.delta   # eq. (4)
        return obj
    
    @classmethod
    def from_delta(cls, delta_ns: int) -> "TimeAligner":
        """
        Build an aligner when you already know Δ (sensor → Aria) in ns.
        """
        obj = cls.__new__(cls)          # bypass __init__
        obj.delta = int(delta_ns)
        return obj

    # ---------- helpers ----------
    def to_aria_time(self, sensor_ts_ns: int) -> int:
        return sensor_ts_ns + self.delta

    def get_delta(self) -> int:
        return self.delta
    
