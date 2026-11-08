"""Instant-by-instant readout of I_Na and I_Kr *while* the ECG is being written.

:mod:`ecg12sim.tracking` answers "what did each current do over the whole beat".
This module answers the narrower, more immediate question: **at this millisecond,
with the ECG at this voltage, what are I_Na and I_Kr actually doing?**

For each sampled instant it reports, for both currents:

``peak``
    the most extreme value reached anywhere in the wall during that reporting
    bin (most negative for inward I_Na, most positive for outward I_Kr).
``mean``
    the average across all sites and across the bin. This is the per-site form
    of the wall-summed signal, and it is the quantity that tracks the ECG
    deflection, because the extracellular potential integrates over the whole
    wall rather than following any single cell.
``on%``
    percentage of sites where the current exceeds 5% of its own global peak
    magnitude, measured at the instant the peak occurred.
``depth``
    transmural depth of the site holding the peak, 0.0 = endocardium,
    1.0 = epicardium. Watching this column march 0 -> 1 during the QRS *is* the
    depolarisation wavefront crossing the wall.

Rows tile the beat: each one reduces every simulation sample since the previous
row. That is deliberate rather than cosmetic. I_Na's wavefront crosses one cell
every ``dx / CV`` ms (0.23 ms for the default strand), so reading a single
instant per row at a 1 ms stride aliases against the cell lattice, making the
reported peak oscillate and fall short of the beat's true maximum.

The core routine :func:`build_timeline` deliberately takes plain arrays plus a
per-site transmural depth rather than a cable object, so the same readout works
for the 1-D strand (depth = cell index / n_cells) and for a 3-D anatomical mesh
(depth = the universal-ventricular-coordinate transmural coordinate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im
from .cable import CableResult

_REGION_LABEL = {im.ENDO: "endo", im.MID: "mid", im.EPI: "epi"}

# A site counts as "participating" above this fraction of the current's own
# global peak magnitude.
ACTIVE_FRACTION_OF_PEAK = 0.05


@dataclass
class CurrentSnapshot:
    """One current's state across the wall at a single instant."""

    peak: float             # pA/pF, signed extreme across sites
    mean: float             # pA/pF, mean across sites
    active_frac: float      # 0-1, fraction of sites appreciably conducting
    peak_depth: float       # 0 = endo, 1 = epi; nan when nothing is active
    peak_region: str        # "endo" / "mid" / "epi", or "-" when inactive


@dataclass
class TimelinePoint:
    """The ECG value and both currents at one instant of the beat."""

    t: float                # ms
    phase: str              # "pre" / "QRS" / "ST" / "T" / "post"
    ecg_mv: float           # the ECG trace being formed, mV
    ina: CurrentSnapshot
    ikr: CurrentSnapshot


@dataclass
class Landmark:
    """A named instant of clinical interest, with both currents at that moment."""

    name: str
    point: TimelinePoint


def classify_phase(t: float, qrs_window: tuple[float, float],
                   t_window: tuple[float, float]) -> str:
    """Label an instant by which part of the ECG it belongs to."""
    if t < qrs_window[0]:
        return "pre"
    if t <= qrs_window[1]:
        return "QRS"
    if t < t_window[0]:
        return "ST"
    if t <= t_window[1]:
        return "T"
    return "post"


def adaptive_sample_times(t: np.ndarray, qrs_window: tuple[float, float],
                          fine_step: float = 1.0, coarse_step: float = 10.0,
                          pad: float = 4.0) -> np.ndarray:
    """Sample finely through depolarisation and coarsely through the rest.

    I_Na is a ~1 ms event on a ~500 ms beat, so a uniform stride either misses
    the sodium spike entirely or produces hundreds of near-identical rows. The
    QRS window (plus a short pad either side) is therefore sampled at
    ``fine_step`` and everything else at ``coarse_step``.
    """
    t0, t1 = float(t[0]), float(t[-1])
    lo = max(t0, qrs_window[0] - pad)
    hi = min(t1, qrs_window[1] + pad)

    coarse = np.arange(t0, t1 + 0.5 * coarse_step, coarse_step)
    fine = np.arange(lo, hi + 0.5 * fine_step, fine_step)
    both = np.unique(np.concatenate([coarse, fine, [t1]]))
    return both[(both >= t0) & (both <= t1)]


def _snapshot(block: np.ndarray, depth: np.ndarray, region: np.ndarray,
              threshold: float, inward: bool) -> CurrentSnapshot:
    """Summarise one current over a reporting bin.

    ``block`` holds every simulation sample falling in the bin, shape
    ``(n_sub, n_sites)``. Reducing over the bin rather than reading a single
    instant inside it matters for I_Na: the wavefront crosses roughly one cell
    every ``dx / CV`` ms, so an instantaneous read at a coarser stride aliases
    against the cell lattice and both under-reports the peak and makes it
    oscillate from row to row. Taking the extreme over the bin reports what the
    current actually reached during that interval, and guarantees the table's
    largest value matches the beat's true global peak.

    ``peak``, ``peak_depth`` and ``active_frac`` are all evaluated at the single
    instant within the bin when the current was strongest, so they describe one
    coherent moment. ``mean`` is the time-average of the spatial mean across the
    bin, which is the quantity that tracks the ECG deflection.
    """
    flat = np.argmin(block) if inward else np.argmax(block)
    k_sub, idx = np.unravel_index(flat, block.shape)
    row = block[k_sub]
    peak = float(row[idx])

    if inward:
        active = row <= -abs(threshold)
    else:
        active = row >= abs(threshold)
    n_active = int(active.sum())

    if n_active == 0:
        peak_depth, peak_region = float("nan"), "-"
    else:
        peak_depth = float(depth[idx])
        peak_region = _REGION_LABEL.get(int(region[idx]), "?")

    return CurrentSnapshot(
        peak=peak + 0.0,                      # normalise -0.0 to 0.0
        mean=float(block.mean()) + 0.0,
        active_frac=n_active / row.size,
        peak_depth=peak_depth,
        peak_region=peak_region,
    )


def build_timeline(t: np.ndarray, ecg_mv: np.ndarray,
                   I_Na: np.ndarray, I_Kr: np.ndarray,
                   depth: np.ndarray, region: np.ndarray,
                   qrs_window: tuple[float, float],
                   t_window: tuple[float, float],
                   sample_times: np.ndarray | None = None) -> list[TimelinePoint]:
    """Build the instant-by-instant readout.

    Parameters
    ----------
    t
        Sample times, shape ``(nt,)``, ms.
    ecg_mv
        The ECG trace being formed on the same time base, shape ``(nt,)``, mV.
    I_Na, I_Kr
        Current fields, shape ``(nt, n_sites)``, pA/pF.
    depth
        Transmural depth per site, shape ``(n_sites,)``, 0 = endo, 1 = epi.
    region
        Region code per site (``ionic_model.ENDO`` / ``MID`` / ``EPI``).
    qrs_window, t_window
        Depolarisation and repolarisation windows, used only to label phases and
        to choose the sampling density.
    sample_times
        Instants to report. Defaults to :func:`adaptive_sample_times`.
    """
    if sample_times is None:
        sample_times = adaptive_sample_times(t, qrs_window)

    # Thresholds come from each current's own global extreme, so "active" means
    # the same thing here as it does in the beat-wide tracking report.
    ina_thr = ACTIVE_FRACTION_OF_PEAK * abs(float(I_Na.min()))
    ikr_thr = ACTIVE_FRACTION_OF_PEAK * abs(float(I_Kr.max()))

    # Map each requested time onto a simulation sample index, de-duplicated and
    # ordered. Each row then reduces the currents over the samples lying between
    # the previous row's index and its own, so consecutive bins tile the beat
    # without gaps and no part of the wavefront is skipped.
    idxs: list[int] = []
    for ts in np.sort(np.asarray(sample_times, dtype=float)):
        k = int(np.argmin(np.abs(t - ts)))
        if not idxs or k != idxs[-1]:
            idxs.append(k)

    rows: list[TimelinePoint] = []
    prev = None
    for k in idxs:
        start = 0 if prev is None else prev + 1
        stop = max(k + 1, start + 1)              # always at least one sample
        sl = slice(start, stop)
        rows.append(TimelinePoint(
            t=float(t[k]),
            phase=classify_phase(float(t[k]), qrs_window, t_window),
            ecg_mv=float(ecg_mv[k]),
            ina=_snapshot(I_Na[sl], depth, region, ina_thr, inward=True),
            ikr=_snapshot(I_Kr[sl], depth, region, ikr_thr, inward=False),
        ))
        prev = k
    return rows


def cable_depth(res: CableResult) -> np.ndarray:
    """Transmural depth (0 = endo, 1 = epi) for each cell of the strand.

    Matches the ``frac = i / (n_cells - 1)`` convention that
    :func:`ecg12sim.cable._region_scales` uses to assign endo/mid/epi.
    """
    n = len(res.x)
    return np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)


def timeline_from_cable(res: CableResult, ecg_mv: np.ndarray,
                        qrs_window: tuple[float, float],
                        t_window: tuple[float, float],
                        step: float | None = None) -> list[TimelinePoint]:
    """Convenience adapter: build the timeline for a 1-D strand result.

    ``step`` forces a uniform stride in ms; the default uses the adaptive
    QRS-fine / elsewhere-coarse sampling.
    """
    if step is None:
        sample_times = None
    else:
        sample_times = np.arange(float(res.t[0]), float(res.t[-1]) + 0.5 * step, step)
    return build_timeline(
        t=res.t, ecg_mv=ecg_mv, I_Na=res.I_Na, I_Kr=res.I_Kr,
        depth=cable_depth(res), region=res.region,
        qrs_window=qrs_window, t_window=t_window, sample_times=sample_times,
    )


def timeline_from_reduced(t: np.ndarray, signal: np.ndarray,
                          ina, ikr,
                          qrs_window: tuple[float, float],
                          t_window: tuple[float, float],
                          step: float | None = None,
                          fine_step: float = 1.0,
                          coarse_step: float = 10.0) -> list[TimelinePoint]:
    """Build the readout from pre-reduced per-sample statistics.

    The mesh model cannot hand over ``(nt, n_sites)`` current fields: at 355k
    nodes those would run to tens of gigabytes. Instead it computes the spatial
    peak, mean, active fraction and peak depth exactly for every sample (see
    :mod:`ecg12sim.mesh_sim`), and this function bins those down to table rows
    using the same tiling rule as :func:`build_timeline`.

    ``ina`` and ``ikr`` need only expose ``peak``, ``mean``, ``active_frac`` and
    ``peak_depth`` arrays, each the same length as ``t``.
    """
    if step is None:
        sample_times = adaptive_sample_times(t, qrs_window, fine_step=fine_step,
                                             coarse_step=coarse_step)
    else:
        sample_times = np.arange(float(t[0]), float(t[-1]) + 0.5 * step, step)

    idxs: list[int] = []
    for ts in np.sort(np.asarray(sample_times, dtype=float)):
        k = int(np.argmin(np.abs(t - ts)))
        if not idxs or k != idxs[-1]:
            idxs.append(k)

    def _snap(src, sl: slice, inward: bool) -> CurrentSnapshot:
        peaks = src.peak[sl]
        rel = int(np.argmin(peaks)) if inward else int(np.argmax(peaks))
        at = sl.start + rel
        return CurrentSnapshot(
            peak=float(src.peak[at]) + 0.0,
            mean=float(np.mean(src.mean[sl])) + 0.0,
            active_frac=float(src.active_frac[at]),
            peak_depth=float(src.peak_depth[at]),
            peak_region="-",       # depth is the reported locator on the mesh
        )

    rows: list[TimelinePoint] = []
    prev = None
    for k in idxs:
        start = 0 if prev is None else prev + 1
        sl = slice(start, max(k + 1, start + 1))
        rows.append(TimelinePoint(
            t=float(t[k]),
            phase=classify_phase(float(t[k]), qrs_window, t_window),
            ecg_mv=float(signal[k]),
            ina=_snap(ina, sl, inward=True),
            ikr=_snap(ikr, sl, inward=False),
        ))
        prev = k
    return rows


def find_landmarks(rows: list[TimelinePoint]) -> list[Landmark]:
    """Pick out the clinically named instants of the beat from a timeline.

    Uses the timeline rows themselves so the reported currents are exactly the
    ones printed in the table.
    """
    if not rows:
        return []

    out: list[Landmark] = []

    def _first(pred):
        return next((r for r in rows if pred(r)), None)

    def _extreme(pred, key, biggest=True):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None
        return max(sub, key=key) if biggest else min(sub, key=key)

    qrs_on = _first(lambda r: r.phase == "QRS")
    if qrs_on:
        out.append(Landmark("QRS onset", qrs_on))

    ina_max = _extreme(lambda r: True, lambda r: -r.ina.peak)
    if ina_max:
        out.append(Landmark("peak I_Na", ina_max))

    r_peak = _extreme(lambda r: r.phase == "QRS", lambda r: abs(r.ecg_mv))
    if r_peak:
        out.append(Landmark("R peak", r_peak))

    j_point = _extreme(lambda r: r.phase == "QRS", lambda r: r.t)
    if j_point:
        out.append(Landmark("QRS end (J)", j_point))

    t_peak = _extreme(lambda r: r.phase in ("ST", "T"), lambda r: abs(r.ecg_mv))
    if t_peak:
        out.append(Landmark("T peak", t_peak))

    # Restricted to after depolarisation on purpose. I_Kr carries a brief
    # transient at the action-potential overshoot (the inactivation gate takes
    # a few ms to close), so an unrestricted search can land on that instead of
    # the repolarisation peak that actually writes the T wave. See
    # ecg12sim.templates.IKR_REPOL_SEARCH_MS.
    ikr_max = _extreme(lambda r: r.phase in ("ST", "T", "post"),
                       lambda r: r.ikr.peak)
    if ikr_max:
        out.append(Landmark("peak I_Kr", ikr_max))

    t_end = _extreme(lambda r: r.phase == "T", lambda r: r.t)
    if t_end:
        out.append(Landmark("T end", t_end))

    out.sort(key=lambda lm: lm.point.t)

    # Several landmarks legitimately coincide (the global I_Na peak often lands
    # on the last QRS sample, and the T peak on the I_Kr peak). Merge them into
    # one row instead of printing identical lines.
    merged: list[Landmark] = []
    for lm in out:
        if merged and merged[-1].point is lm.point:
            merged[-1] = Landmark(f"{merged[-1].name} + {lm.name}", lm.point)
        else:
            merged.append(lm)
    return merged


def _fmt_depth(s: CurrentSnapshot) -> str:
    if not np.isfinite(s.peak_depth):
        return "    -"
    return f"{s.peak_depth:5.2f}"


_HEADER = (
    f"{'t':>6s} {'phase':>5s} {'signal':>7s} |"
    f"{'I_Na peak':>10s} {'mean':>8s} {'on%':>6s} {'depth':>6s} |"
    f"{'I_Kr peak':>10s} {'mean':>8s} {'on%':>6s} {'depth':>6s}"
)
_UNITS = (
    f"{'ms':>6s} {'':>5s} {'mV':>7s} |"
    f"{'pA/pF':>10s} {'pA/pF':>8s} {'':>6s} {'':>6s} |"
    f"{'pA/pF':>10s} {'pA/pF':>8s} {'':>6s} {'':>6s}"
)
_RULE = "-" * len(_HEADER)


def _fmt_row(r: TimelinePoint) -> str:
    return (
        f"{r.t:6.1f} {r.phase:>5s} {r.ecg_mv:7.3f} |"
        f"{r.ina.peak:10.1f} {r.ina.mean:8.2f} "
        f"{r.ina.active_frac * 100:5.1f}% {_fmt_depth(r.ina):>6s} |"
        f"{r.ikr.peak:10.3f} {r.ikr.mean:8.3f} "
        f"{r.ikr.active_frac * 100:5.1f}% {_fmt_depth(r.ikr):>6s}"
    )


def format_timeline(rows: list[TimelinePoint],
                    signal_label: str = "lead II") -> str:
    """Render the instant-by-instant readout as a text table."""
    L = [
        f"I_Na / I_Kr as the ECG is formed (signal column = {signal_label})",
        "peak/mean reduce every sample since the previous row; signal is the value at t.",
        "depth: 0.00 = endocardium, 1.00 = epicardium, site holding that bin's peak",
        _RULE,
        _HEADER,
        _UNITS,
        _RULE,
    ]
    last_phase = None
    for r in rows:
        if last_phase is not None and r.phase != last_phase:
            L.append(_RULE)           # visual break between ECG phases
        L.append(_fmt_row(r))
        last_phase = r.phase
    return "\n".join(L)


def format_landmarks(marks: list[Landmark],
                     signal_label: str = "lead II") -> str:
    """Render just the named instants of the beat."""
    L = [
        f"Both currents at the named instants of the beat (signal = {signal_label})",
        _RULE,
        f"{'instant':>23s} {'t':>7s} {'signal':>7s} |"
        f"{'I_Na peak':>10s} {'@depth':>7s} {'on%':>6s} |"
        f"{'I_Kr peak':>10s} {'@depth':>7s} {'on%':>6s}",
        f"{'':>23s} {'ms':>7s} {'mV':>7s} |"
        f"{'pA/pF':>10s} {'':>7s} {'':>6s} |"
        f"{'pA/pF':>10s} {'':>7s} {'':>6s}",
        _RULE,
    ]
    for lm in marks:
        p = lm.point
        L.append(
            f"{lm.name:>23s} {p.t:7.1f} {p.ecg_mv:7.3f} |"
            f"{p.ina.peak:10.1f} {_fmt_depth(p.ina):>7s} "
            f"{p.ina.active_frac * 100:5.1f}% |"
            f"{p.ikr.peak:10.3f} {_fmt_depth(p.ikr):>7s} "
            f"{p.ikr.active_frac * 100:5.1f}%"
        )
    return "\n".join(L)
