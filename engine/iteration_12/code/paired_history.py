"""Finite-mixture histories using event gaps and paired absolute timestamps.

Independent numerical representation amendment. No event is jittered or merged.
Only full-horizon scoring from an empty initial history is supported here.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "iteration_5/code"))
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from empirical_process import EmpiricalHistory, stress_marks, G6, unit_productivity


def difference(hi, lo, other_hi, other_lo):
    """Correctly sum the four represented components; no absolute-time jitter."""
    return math.fsum((float(hi), -float(other_hi), float(lo), -float(other_lo)))


class PairedHistory(EmpiricalHistory):
    def __init__(self, times, excess, beta, end, coordinate_horizon=24394., start=0.,
                 order=16, memory_limit_mb=1600., *, times_lo=None, event_gaps=None):
        if start != 0:
            raise ValueError("Paired continuation supports full-window start=0 only")
        self.times = np.asarray(times, dtype=float)
        self.times_lo = np.zeros_like(self.times) if times_lo is None else np.asarray(times_lo, dtype=float)
        self.excess = np.asarray(excess, dtype=float)
        if (self.times.ndim != 1 or not len(self.times) or
                self.times.shape != self.excess.shape or self.times.shape != self.times_lo.shape or
                not all(np.isfinite(a).all() for a in (self.times, self.times_lo, self.excess)) or
                np.any(self.excess < 0) or not np.isfinite(beta) or beta <= 0 or
                not 0 < end or coordinate_horizon <= 0 or int(order) <= 0):
            raise ValueError("Invalid paired catalogue/window")
        represented = np.array([difference(self.times[j], self.times_lo[j],
                                self.times[j-1] if j else 0., self.times_lo[j-1] if j else 0.)
                                for j in range(len(self.times))])
        if np.any(represented <= 0):
            raise ValueError("Every represented event gap must be positive")
        self.event_gaps = represented if event_gaps is None else np.asarray(event_gaps, dtype=float)
        if self.event_gaps.shape != represented.shape or not np.isfinite(self.event_gaps).all() or np.any(self.event_gaps <= 0):
            raise ValueError("Invalid authoritative positive gaps")
        if any(abs(float(lo)) > 4*math.ulp(float(hi)) for hi, lo in zip(self.times, self.times_lo)):
            raise ValueError("Noncanonical coarse/residual clock representation")
        rounding_allowance = np.array([4*(math.ulp(float(lo))+
                                       math.ulp(float(self.times_lo[j-1] if j else 0.))+
                                       math.ulp(float(represented[j]))+math.ulp(float(self.event_gaps[j])))
                                       for j, lo in enumerate(self.times_lo)])
        if np.any(abs(represented-self.event_gaps) > rounding_allowance):
            raise ValueError("Authoritative gaps inconsistent with paired clock rounding")
        # A two-double absolute representation rounds the exact Decimal clock.
        # Its discrepancy from separately rounded authoritative adjacent gaps
        # is measured, not repaired by moving event times.
        self.max_pair_gap_absolute_difference = float(np.max(abs(represented-self.event_gaps)))
        self.max_pair_gap_relative_difference = float(np.max(abs(represented-self.event_gaps)/self.event_gaps))
        remaining = difference(end, 0., self.times[-1], self.times_lo[-1])
        if remaining <= 0:
            raise ValueError("Catalogue must end strictly before horizon")
        self.end, self.start, self.beta = float(end), 0., float(beta)
        self.coordinate_horizon, self.order = float(coordinate_horizon), int(order)
        self.memory_limit_mb = float(memory_limit_mb)
        self.event_mask = np.ones(len(self.times), dtype=bool)
        self.xe = 2*(self.times+self.times_lo)/self.coordinate_horizon-1
        self.stress, self.trigger = {}, {}
        offsets = np.geomspace(1e-6, 1e5, 23)
        offsets = np.sort(np.r_[offsets, np.sqrt(offsets[:-1]*offsets[1:])])
        intervals = np.r_[self.event_gaps, remaining]
        total_gap_error = abs(math.fsum(intervals)-self.end)
        total_allowance = math.fsum(rounding_allowance)+4*math.ulp(self.end)
        if total_gap_error > total_allowance:
            raise ValueError("Interval durations do not cover the full horizon")
        self.interval_duration_sum_error = total_gap_error
        panels = sum(1+int(np.count_nonzero(offsets < gap)) for gap in intervals)
        nodes = panels*self.order
        self.estimated_memory_mb = (24*len(self.times)**2+nodes*8*35)/1024**2
        if self.estimated_memory_mb > self.memory_limit_mb:
            raise MemoryError(f"paired history estimate {self.estimated_memory_mb:.3f}MiB exceeds allowance")
        z, w = leggauss(self.order)
        self.qindex = np.empty(nodes, dtype=np.int64)
        self.qlag, self.wq, self.xq, self.tq = [np.empty(nodes) for _ in range(4)]
        position = 0
        for j, gap in enumerate(intervals):
            edges = np.r_[0., offsets[offsets < gap], gap]
            rel = ((edges[:-1, None]+edges[1:, None])/2 +
                   (edges[1:, None]-edges[:-1, None])*z/2).ravel()
            weight = ((edges[1:, None]-edges[:-1, None])*w/2).ravel()
            left_hi = self.times[j-1] if j else 0.
            left_lo = self.times_lo[j-1] if j else 0.
            sl = slice(position, position+len(rel))
            self.qindex[sl], self.qlag[sl], self.wq[sl] = j, rel, weight
            self.tq[sl] = left_hi+(left_lo+rel)  # display/background coordinate only
            self.xq[sl] = 2*self.tq[sl]/self.coordinate_horizon-1
            position += len(rel)
        assert position == nodes and np.all(self.qlag > 0) and np.all(self.wq > 0)

    def event_states(self, amplitudes, kernel):
        weights, rates = kernel.mixture
        result = np.empty((len(self.times), len(rates)))
        state = np.zeros(len(rates))
        for j, (gap, amplitude) in enumerate(zip(self.event_gaps, amplitudes)):
            state *= np.exp(-rates*gap)
            result[j] = state
            state += amplitude
        return result

    def stress_features(self, kernel):
        if kernel.key in self.stress:
            return self.stress[kernel.key]
        if kernel.kind not in ("exp", "ml"):
            raise ValueError("Paired continuation supports finite-mixture stress kernels only")
        gm = stress_marks(self.excess)/G6
        weights, rates = kernel.mixture
        pre = self.event_states(gm, kernel)
        post = np.vstack((np.zeros(len(rates)), pre+gm[:, None]))
        ge = pre@weights
        gq = np.empty(len(self.qlag))
        for a in range(0, len(gq), 2048):
            sl = slice(a, a+2048)
            gq[sl] = (post[self.qindex[sl]]*np.exp(-self.qlag[sl, None]*rates))@weights
        self.stress[kernel.key] = ge, gq
        return ge, gq

    def trigger_features(self, kernel, ratio):
        key = kernel.key, float(ratio)
        if key in self.trigger:
            return self.trigger[key]
        if kernel.kind not in ("exp", "ml"):
            raise ValueError("Paired continuation supports finite-mixture triggering only")
        kp = unit_productivity(self.excess, self.beta, ratio)
        weights, rates = kernel.mixture
        he = self.event_states(kp, kernel)@(weights*rates)
        durations = np.array([difference(self.end, 0., t, lo) for t, lo in zip(self.times, self.times_lo)])
        integral = float(np.dot(kp, kernel.cdf(durations)))
        self.trigger[key] = he, integral
        return he, integral


def cuda_history_class():
    """Import CUDA only when an actual fit is requested."""
    sys.path.insert(0, str(ROOT / "iteration_5_performance/gpu"))
    from cuda_history import CUDAHistory
    import torch

    class PairedCUDAHistory(PairedHistory):
        tensor = CUDAHistory.tensor
        objective = CUDAHistory.objective

        def __init__(self, *args, **kwargs):
            if not torch.cuda.is_available():
                raise RuntimeError("Usable CUDA device required")
            super().__init__(*args, **kwargs)
            self.device = torch.device("cuda")
            self.gpu_stress, self.gpu_objectives = {}, {}
            self.qindex_t = torch.as_tensor(self.qindex, dtype=torch.int64, device=self.device)
            for name in ("qlag", "xq", "wq", "xe"):
                setattr(self, name+"_t", self.tensor(getattr(self, name)))

        def stress_features(self, kernel):
            value = PairedHistory.stress_features(self, kernel)
            if kernel.key not in self.gpu_stress:
                self.gpu_stress[kernel.key] = tuple(self.tensor(x) for x in value)
            return value

    return PairedCUDAHistory
