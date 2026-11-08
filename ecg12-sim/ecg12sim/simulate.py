"""High-level orchestration: run single-cell current traces and the 12-lead ECG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im
from . import leads as ld
from .cable import CableConfig, CableResult, simulate_cable

# transmural heterogeneity presets for an isolated representative cell
_CELL_SCALES = {
    "endo": dict(gks_scale=1.0, gto_scale=0.25),
    "mid": dict(gks_scale=0.25, gto_scale=1.0),
    "epi": dict(gks_scale=1.0, gto_scale=1.0),
}


@dataclass
class CellTrace:
    t: np.ndarray
    V: np.ndarray
    I_Na: np.ndarray
    I_Kr: np.ndarray
    I_Ks: np.ndarray
    I_to: np.ndarray
    I_CaL: np.ndarray
    I_K1: np.ndarray
    apd90: float
    cell_type: str


def run_single_cell(drug: im.Drug, cell_type: str = "epi",
                    duration: float = 500.0, dt: float = 0.02,
                    cond: im.Conductances | None = None) -> CellTrace:
    """Simulate one paced ventricular cell and record its current time-courses."""
    cond = cond or im.Conductances()
    sc = _CELL_SCALES[cell_type]
    n = int(round(duration / dt))
    s = im.initial_state(1)
    t = np.arange(n) * dt

    V = np.zeros(n)
    rec = {k: np.zeros(n) for k in ("I_Na", "I_Kr", "I_Ks", "I_to", "I_CaL", "I_K1")}
    for k in range(n):
        stim = 52.0 if (5.0 <= t[k] < 6.0) else 0.0
        s, cur = im.step(s, dt, stim, cond, drug,
                         gto_scale=sc["gto_scale"], gks_scale=sc["gks_scale"])
        V[k] = s[0, 0]
        for key in rec:
            rec[key][k] = cur[key][0]

    vrest, peak = V[0], V.max()
    v90 = peak - 0.9 * (peak - vrest)
    up = np.argmax(V > -40.0)
    below = np.where(V[up:] < v90)[0]
    apd90 = float(below[0] * dt) if len(below) else float("nan")

    return CellTrace(t=t, V=V, apd90=apd90, cell_type=cell_type,
                     **{k: rec[k] for k in rec})


@dataclass
class EcgResult:
    cable: CableResult
    leads: dict
    qt: float
    gain: float


def run_12lead(drug: im.Drug, cfg: CableConfig | None = None,
               cond: im.Conductances | None = None,
               gain: float | None = None) -> EcgResult:
    """Run the transmural strand and synthesise the 12-lead ECG."""
    cfg = cfg or CableConfig()
    res = simulate_cable(cfg, drug, cond=cond, record_every=5)
    leads, used_gain = ld.synthesize_leads(res, gain=gain)
    return EcgResult(cable=res, leads=leads, qt=ld.qt_interval(res), gain=used_gain)
