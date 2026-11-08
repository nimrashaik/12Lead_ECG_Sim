"""Single-cell action potential, and how it responds to scaling I_Na or I_Kr.

Two things live here:

1. The **complete** action potential of one ventricular myocyte, with every
   current the model carries, not just the two this project tracks.
2. A **sweep** of each tracked conductance across levels spanning well below
   and well above baseline, one current at a time, so the two mechanisms can be
   read off separately.

Why scale rather than block
---------------------------

:meth:`ecg12sim.ionic_model.Drug.from_block_percent` only maps 0-100% block onto
scales at or below 1.0, so it cannot express an *increase*. The underlying
``Drug`` fields are plain multiplicative factors, so this module drives them
directly and can go above baseline as well as below. A scale of 1.0 is
untouched, 0.5 is half the conductance (a 50% block), 2.0 is double.

Expected directions, which double as a sanity check on the sweep:

``I_Na``
    scales the phase-0 upstroke. More sodium conductance gives a faster
    ``dV/dt max`` and a higher overshoot; less gives a slower, smaller upstroke,
    and far enough down the cell stops producing a real action potential at all.
    APD is only weakly affected.
``I_Kr``
    scales repolarisation. More gives a shorter APD; less prolongs it, which is
    the mechanism behind drug-induced QT prolongation. The upstroke is
    essentially untouched.

That asymmetry is the point: the two currents act on orthogonal features of the
same waveform.

Vectorisation
-------------

``ionic_model`` is vectorised over cells and the ``Drug`` scale factors
broadcast, so an entire sweep runs as **one** ensemble integration with a
per-cell scale rather than one integration per level. Running them separately
costs roughly ``n_levels`` times as much for an identical answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im
from .templates import _PREPACE_MS, IKR_REPOL_SEARCH_MS

#: Ten levels on a sqrt(2) ladder, spanning 0.18x to 4x baseline and including
#: baseline exactly (exponent 0). Geometric spacing rather than linear so that
#: a halving and a doubling are equally far from baseline.
SWEEP_SCALES = tuple(float(2.0 ** e) for e in np.arange(-2.5, 2.5, 0.5))

#: Currents recorded for the complete-action-potential view.
ALL_CURRENTS = ("I_Na", "I_CaL", "I_to", "I_Kr", "I_Ks", "I_K1",
                "I_pK", "I_NaK", "I_bNa")

_CELL_SCALES = {
    "endo": dict(gks_scale=1.0, gto_scale=0.25),
    "mid": dict(gks_scale=0.25, gto_scale=1.0),
    "epi": dict(gks_scale=1.0, gto_scale=1.0),
}

#: A depolarisation smaller than this is not treated as an action potential.
_AP_MIN_AMPLITUDE = 20.0      # mV


@dataclass
class APMetrics:
    """Scalar descriptors of one action potential."""

    scale: float
    excited: bool
    v_rest: float          # mV
    v_peak: float          # mV
    dvdt_max: float        # mV/ms, the classic I_Na readout
    apd50: float           # ms
    apd90: float           # ms
    v_plateau: float       # mV, 100 ms after the upstroke
    ina_peak: float        # pA/pF (inward, negative)
    ikr_peak: float        # pA/pF, repolarisation peak
    t_ikr_peak: float      # ms


@dataclass
class SweepResult:
    """One conductance swept across levels, with traces and metrics."""

    which: str                  # "I_Na" or "I_Kr"
    cell_type: str
    t: np.ndarray               # (nt,) ms
    scales: np.ndarray          # (n_levels,)
    Vm: np.ndarray              # (n_levels, nt) mV
    I_Na: np.ndarray            # (n_levels, nt) pA/pF
    I_Kr: np.ndarray            # (n_levels, nt) pA/pF
    metrics: list               # list[APMetrics], same order as scales

    @property
    def baseline_index(self) -> int:
        return int(np.argmin(np.abs(self.scales - 1.0)))


def _run_ensemble(drug: im.Drug, cond: im.Conductances, n_cells: int,
                  cell_type: str, duration: float, dt: float,
                  prepace: float, record: tuple):
    """Integrate ``n_cells`` cells that differ only in their drug scaling."""
    sc = _CELL_SCALES[cell_type]
    gto, gks = sc["gto_scale"], sc["gks_scale"]

    s = im.initial_state(n_cells)
    zero = np.zeros(n_cells)
    # Relax to the true resting state *under the altered conductance*, so a
    # scaled cell starts from its own equilibrium rather than the baseline one.
    for _ in range(int(round(prepace / dt))):
        s, _ = im.step(s, dt, zero, cond, drug,
                       gto_scale=gto, gks_scale=gks)

    n = int(round(duration / dt))
    V = np.zeros((n, n_cells))
    rec = {k: np.zeros((n, n_cells)) for k in record}
    stim = np.full(n_cells, 52.0)

    for k in range(n):
        on = (k * dt) < 1.0
        s, cur = im.step(s, dt, stim if on else zero, cond, drug,
                         gto_scale=gto, gks_scale=gks)
        V[k] = s[0]
        for key in rec:
            rec[key][k] = cur[key]
    return V, rec


def _metrics(t: np.ndarray, V: np.ndarray, ina: np.ndarray, ikr: np.ndarray,
             scale: float, dt: float) -> APMetrics:
    """Measure one action potential."""
    v_rest = float(V[0])
    v_peak = float(V.max())
    dvdt = np.diff(V) / dt
    dvdt_max = float(dvdt.max())

    excited = (v_peak - v_rest) >= _AP_MIN_AMPLITUDE
    i_up = int(np.argmax(dvdt))                 # steepest point of the upstroke
    i_pk = int(np.argmax(V))

    def _apd(frac: float) -> float:
        if not excited:
            return float("nan")
        v_x = v_peak - frac * (v_peak - v_rest)
        below = np.flatnonzero(V[i_pk:] < v_x)
        if below.size == 0:
            return float("nan")
        return float(t[i_pk + below[0]] - t[i_up])

    plateau_idx = min(i_up + int(round(100.0 / dt)), V.size - 1)

    # I_Kr's repolarisation peak, skipping the brief transient at the overshoot.
    late = t >= (t[i_up] + IKR_REPOL_SEARCH_MS)
    if late.any():
        sub = np.where(late, ikr, -np.inf)
        i_kr = int(np.argmax(sub))
    else:
        i_kr = int(np.argmax(ikr))

    return APMetrics(
        scale=float(scale), excited=bool(excited),
        v_rest=v_rest, v_peak=v_peak, dvdt_max=dvdt_max,
        apd50=_apd(0.50), apd90=_apd(0.90),
        v_plateau=float(V[plateau_idx]),
        ina_peak=float(ina.min()),
        ikr_peak=float(ikr[i_kr]), t_ikr_peak=float(t[i_kr]),
    )


def run_ap_sweep(which: str = "I_Na", scales=SWEEP_SCALES,
                 cell_type: str = "epi", duration: float = 500.0,
                 dt: float = 0.02, prepace: float = _PREPACE_MS,
                 cond: im.Conductances | None = None) -> SweepResult:
    """Scale one conductance across ``scales`` and measure the result.

    ``which`` is ``"I_Na"`` or ``"I_Kr"``. The other current is left at
    baseline, so the two mechanisms stay separable.
    """
    if which not in ("I_Na", "I_Kr"):
        raise ValueError(f"which must be 'I_Na' or 'I_Kr', got {which!r}")
    if cell_type not in _CELL_SCALES:
        raise ValueError(f"cell_type must be one of {sorted(_CELL_SCALES)}")

    cond = cond or im.Conductances()
    arr = np.asarray(scales, dtype=float)
    ones = np.ones_like(arr)
    drug = (im.Drug(ina_scale=arr, ikr_scale=ones) if which == "I_Na"
            else im.Drug(ina_scale=ones, ikr_scale=arr))

    V, rec = _run_ensemble(drug, cond, arr.size, cell_type, duration, dt,
                           prepace, ("I_Na", "I_Kr"))
    t = np.arange(V.shape[0]) * dt

    mets = [_metrics(t, V[:, i], rec["I_Na"][:, i], rec["I_Kr"][:, i],
                     arr[i], dt) for i in range(arr.size)]

    return SweepResult(which=which, cell_type=cell_type, t=t, scales=arr,
                       Vm=V.T.copy(), I_Na=rec["I_Na"].T.copy(),
                       I_Kr=rec["I_Kr"].T.copy(), metrics=mets)


def run_complete_ap(cell_type: str = "epi", duration: float = 500.0,
                    dt: float = 0.02, prepace: float = _PREPACE_MS,
                    cond: im.Conductances | None = None):
    """Baseline action potential with every current the model carries.

    Returns ``(t, Vm, {current_name: trace}, APMetrics)``.
    """
    cond = cond or im.Conductances()
    drug = im.Drug()
    V, rec = _run_ensemble(drug, cond, 1, cell_type, duration, dt, prepace,
                           ALL_CURRENTS)
    t = np.arange(V.shape[0]) * dt
    traces = {k: rec[k][:, 0].copy() for k in ALL_CURRENTS}
    met = _metrics(t, V[:, 0], traces["I_Na"], traces["I_Kr"], 1.0, dt)
    return t, V[:, 0].copy(), traces, met


# ---------------------------------------------------------------------------
# Numeric reporting
# ---------------------------------------------------------------------------

def format_sweep_table(res: SweepResult) -> str:
    """Render the sweep as a numeric table, with change from baseline."""
    b = res.metrics[res.baseline_index]
    label = res.which.replace("I_", "I_")

    L = [f"{label} scaled across {len(res.metrics)} levels "
         f"({res.cell_type} cell, the other current left at baseline)",
         "scale 1.00 = untouched; 0.50 = 50% block; 2.00 = doubled",
         "-" * 92,
         f"{'scale':>6s} {'dV/dt max':>10s} {'V peak':>8s} {'APD50':>8s} "
         f"{'APD90':>8s} {'dAPD90':>8s} {'I_Na pk':>9s} {'I_Kr pk':>8s} "
         f"{'plateau':>8s}",
         f"{'':>6s} {'mV/ms':>10s} {'mV':>8s} {'ms':>8s} "
         f"{'ms':>8s} {'%':>8s} {'pA/pF':>9s} {'pA/pF':>8s} {'mV':>8s}",
         "-" * 92]

    for m in res.metrics:
        if not m.excited:
            L.append(f"{m.scale:6.2f} {m.dvdt_max:10.1f} {m.v_peak:8.1f} "
                     f"{'--':>8s} {'--':>8s} {'--':>8s} "
                     f"{m.ina_peak:9.1f} {m.ikr_peak:8.3f} "
                     f"{m.v_plateau:8.1f}   (no action potential)")
            continue
        d_apd = (100.0 * (m.apd90 - b.apd90) / b.apd90
                 if np.isfinite(b.apd90) and b.apd90 else float("nan"))
        star = "  <- baseline" if m is b else ""
        L.append(f"{m.scale:6.2f} {m.dvdt_max:10.1f} {m.v_peak:8.1f} "
                 f"{m.apd50:8.1f} {m.apd90:8.1f} {d_apd:+8.1f} "
                 f"{m.ina_peak:9.1f} {m.ikr_peak:8.3f} "
                 f"{m.v_plateau:8.1f}{star}")

    ok = [m for m in res.metrics if m.excited]
    L.append("-" * 92)
    if ok:
        L.append(f"{'span':>6s} "
                 f"{max(m.dvdt_max for m in ok) - min(m.dvdt_max for m in ok):10.1f} "
                 f"{max(m.v_peak for m in ok) - min(m.v_peak for m in ok):8.1f} "
                 f"{max(m.apd50 for m in ok) - min(m.apd50 for m in ok):8.1f} "
                 f"{max(m.apd90 for m in ok) - min(m.apd90 for m in ok):8.1f}")
    n_fail = sum(1 for m in res.metrics if not m.excited)
    if n_fail:
        L.append(f"{n_fail} level(s) failed to produce an action potential "
                 f"(depolarisation under {_AP_MIN_AMPLITUDE:.0f} mV).")
    return "\n".join(L)


def format_complete_ap(t, V, traces, met: APMetrics, cell_type: str) -> str:
    """Render the baseline action potential and every current numerically."""
    L = [f"Complete action potential, {cell_type} ventricular myocyte",
         "-" * 68,
         f"  resting potential     : {met.v_rest:8.2f} mV",
         f"  peak / overshoot      : {met.v_peak:8.2f} mV",
         f"  dV/dt max (upstroke)  : {met.dvdt_max:8.1f} mV/ms",
         f"  APD50 / APD90         : {met.apd50:8.1f} / {met.apd90:.1f} ms",
         f"  plateau (+100 ms)     : {met.v_plateau:8.2f} mV",
         "",
         f"{'current':>8s} {'peak':>10s} {'t_peak':>8s} {'charge':>11s}",
         f"{'':>8s} {'pA/pF':>10s} {'ms':>8s} {'pA/pF.ms':>11s}",
         "-" * 68]
    for name in ALL_CURRENTS:
        tr = traces[name]
        i = int(np.argmax(np.abs(tr)))
        L.append(f"{name:>8s} {tr[i]:10.3f} {t[i]:8.1f} "
                 f"{np.trapezoid(tr, t):11.1f}")
    L.append("-" * 68)
    L.append("Peak is the largest excursion either way; charge is the time "
             "integral (net ion movement).")
    return "\n".join(L)
