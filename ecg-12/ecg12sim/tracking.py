"""Track I_Na and I_Kr through a normal beat and relate them to the ECG.

Nothing is blocked or altered here. The purpose is observational: during one
ordinary heartbeat, measure *where* and *when* each current flows and tie that to
the ECG waveform the same simulation produces.

What the two currents do, and what the tracking therefore shows:

``I_Na``
    Large (hundreds of pA/pF) but very brief. It fires only in cells that are
    actively depolarising, so in space-time it is a narrow diagonal stripe
    sweeping endocardium -> epicardium. The slope of that stripe *is* the
    conduction velocity, and the stripe coincides with the QRS complex.

``I_Kr``
    Small (~1 pA/pF) but long-lasting. It grows through the plateau and peaks as
    each cell repolarises. Because M cells repolarise last, the I_Kr peak is
    spread out across the wall, and that dispersion is what writes the T wave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im
from .cable import CableResult

_REGION_LABEL = {im.ENDO: "endo", im.MID: "mid (M)", im.EPI: "epi"}


@dataclass
class RegionCurrentStats:
    """Per-region summary of one current's behaviour during the beat."""

    label: str
    peak: float          # pA/pF (signed: negative for inward I_Na)
    t_peak: float        # ms, time of that peak
    charge: float        # pA/pF * ms, time-integral (net ion movement)


@dataclass
class TrackingReport:
    conduction_velocity: float          # cm/s, from the I_Na wavefront
    qrs_window: tuple[float, float]     # ms, depolarisation phase
    t_window: tuple[float, float]       # ms, repolarisation phase
    ina: list[RegionCurrentStats]
    ikr: list[RegionCurrentStats]
    ina_peak_global: float
    ikr_peak_global: float
    ina_active_fraction: float          # fraction of the beat I_Na is appreciably on
    ikr_active_fraction: float


def _region_stats(t: np.ndarray, field: np.ndarray, region: np.ndarray,
                  inward: bool) -> list[RegionCurrentStats]:
    """Summarise a current field region by region.

    Each cell's *own* peak and peak time are measured first, then averaged over
    the cells of the region. Averaging the traces first would be wrong for I_Na:
    neighbouring cells fire at different times, so a mean trace smears a narrow
    1 ms spike into something an order of magnitude too small.
    """
    out = []
    for code in (im.ENDO, im.MID, im.EPI):
        sel = region == code
        if not np.any(sel):
            continue
        sub = field[:, sel]                                  # (nt, n_sel)
        idx = np.argmin(sub, axis=0) if inward else np.argmax(sub, axis=0)
        peaks = sub[idx, np.arange(sub.shape[1])]
        out.append(RegionCurrentStats(
            label=_REGION_LABEL[code],
            peak=float(peaks.mean()),
            t_peak=float(t[idx].mean()),
            charge=float(np.trapezoid(sub, t, axis=0).mean()),
        ))
    return out


def conduction_velocity(res: CableResult) -> float:
    """Conduction velocity (cm/s) from the slope of the I_Na activation wavefront.

    Fitted over the middle 60% of the wall to avoid stimulus and boundary effects.
    """
    act = res.activation_time
    x = res.x
    n = len(x)
    lo, hi = int(0.2 * n), int(0.8 * n)
    xs, ts = x[lo:hi], act[lo:hi]
    ok = ~np.isnan(ts)
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(ts[ok], xs[ok], 1)[0]   # cm per ms
    return float(slope * 1000.0)               # cm/s


def ecg_windows(res: CableResult) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the (QRS, T) time windows implied by activation/repolarisation."""
    qrs = (float(np.nanmin(res.activation_time)),
           float(np.nanmax(res.activation_time)))
    tw = (float(np.nanmin(res.repol_time)), float(np.nanmax(res.repol_time)))
    return qrs, tw


def analyse(res: CableResult) -> TrackingReport:
    """Build the full tracking report for one beat."""
    t = res.t
    qrs, tw = ecg_windows(res)

    ina_peak = float(res.I_Na.min())        # inward -> most negative
    ikr_peak = float(res.I_Kr.max())        # outward -> most positive

    # "appreciably on" = any cell above 5% of that current's own peak magnitude
    ina_on = (res.I_Na.min(axis=1) < 0.05 * ina_peak).mean()
    ikr_on = (res.I_Kr.max(axis=1) > 0.05 * ikr_peak).mean()

    return TrackingReport(
        conduction_velocity=conduction_velocity(res),
        qrs_window=qrs,
        t_window=tw,
        ina=_region_stats(t, res.I_Na, res.region, inward=True),
        ikr=_region_stats(t, res.I_Kr, res.region, inward=False),
        ina_peak_global=ina_peak,
        ikr_peak_global=ikr_peak,
        ina_active_fraction=float(ina_on),
        ikr_active_fraction=float(ikr_on),
    )


def format_report(rep: TrackingReport, qt: float) -> str:
    """Render the tracking report as a readable text table."""
    L = []
    L.append("Beat-wide tracking of I_Na and I_Kr (no channel block applied)")
    L.append("-" * 62)
    L.append(f"Conduction velocity (from I_Na wavefront) : {rep.conduction_velocity:.1f} cm/s")
    L.append(f"QRS / depolarisation window               : "
             f"{rep.qrs_window[0]:.1f} - {rep.qrs_window[1]:.1f} ms")
    L.append(f"T wave / repolarisation window            : "
             f"{rep.t_window[0]:.1f} - {rep.t_window[1]:.1f} ms")
    L.append(f"QT interval                               : {qt:.0f} ms")
    L.append("")
    L.append(f"I_Na peak (inward)  : {rep.ina_peak_global:8.1f} pA/pF   "
             f"active {rep.ina_active_fraction * 100:.1f}% of the beat")
    L.append(f"I_Kr peak (outward) : {rep.ikr_peak_global:8.3f} pA/pF   "
             f"active {rep.ikr_active_fraction * 100:.1f}% of the beat")
    L.append("")
    L.append(f"{'region':9s} {'I_Na peak':>11s} {'t_peak':>8s} | "
             f"{'I_Kr peak':>10s} {'t_peak':>8s} {'I_Kr charge':>12s}")
    for a, k in zip(rep.ina, rep.ikr):
        L.append(f"{a.label:9s} {a.peak:11.1f} {a.t_peak:8.1f} | "
                 f"{k.peak:10.3f} {k.t_peak:8.1f} {k.charge:12.1f}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Transmural variation, as numbers
# ---------------------------------------------------------------------------

@dataclass
class DepthBand:
    """Both currents summarised over one band of transmural depth."""

    depth_lo: float
    depth_hi: float
    n_sites: int
    ina_peak: float        # pA/pF, mean of each site's own peak
    ina_t_peak: float      # ms, mean of each site's own peak time
    ikr_peak: float
    ikr_t_peak: float


def band_profile(depth: np.ndarray,
                 ina_peak: np.ndarray, ina_t: np.ndarray,
                 ikr_peak: np.ndarray, ikr_t: np.ndarray,
                 weights: np.ndarray | None = None,
                 n_bands: int = 9) -> list[DepthBand]:
    """Bin per-site current statistics into bands of transmural depth.

    Model-agnostic on purpose: it takes per-site arrays plus a depth coordinate,
    so the same banding serves the 1-D strand (depth = cell index / n_cells) and
    the 3-D mesh (depth = the universal transmural coordinate).

    Each site's *own* peak is measured before averaging, never the other way
    round. Averaging traces first would smear I_Na's ~1 ms spike across the
    stagger between neighbouring sites and under-report it by an order of
    magnitude.
    """
    depth = np.asarray(depth, dtype=float)
    w = np.ones_like(depth) if weights is None else np.asarray(weights, float)
    edges = np.linspace(0.0, 1.0, n_bands + 1)

    out: list[DepthBand] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (depth >= lo) & (depth < hi) if hi < 1.0 else \
              (depth >= lo) & (depth <= hi)
        sel = sel & np.isfinite(ina_t) & np.isfinite(ikr_t) & (w > 0)
        if not sel.any():
            continue
        ww = w[sel]
        tot = ww.sum()
        out.append(DepthBand(
            depth_lo=float(lo), depth_hi=float(hi), n_sites=int(sel.sum()),
            ina_peak=float((ina_peak[sel] * ww).sum() / tot),
            ina_t_peak=float((ina_t[sel] * ww).sum() / tot),
            ikr_peak=float((ikr_peak[sel] * ww).sum() / tot),
            ikr_t_peak=float((ikr_t[sel] * ww).sum() / tot),
        ))
    return out


def cable_band_profile(res: CableResult, n_bands: int = 9) -> list[DepthBand]:
    """Transmural profile of both currents along the 1-D strand."""
    t = res.t
    cols = np.arange(res.I_Na.shape[1])

    i_na = np.argmin(res.I_Na, axis=0)
    i_kr = np.argmax(res.I_Kr, axis=0)
    depth = (np.linspace(0.0, 1.0, len(res.x)) if len(res.x) > 1
             else np.zeros(1))

    return band_profile(
        depth=depth,
        ina_peak=res.I_Na[i_na, cols], ina_t=t[i_na],
        ikr_peak=res.I_Kr[i_kr, cols], ikr_t=t[i_kr],
        n_bands=n_bands)


def format_band_profile(bands: list[DepthBand], title: str = "") -> str:
    """Render the transmural profile, with the spread across the wall."""
    if not bands:
        return "no transmural bands to report"

    L = [title or "Transmural variation of I_Na and I_Kr",
         "depth 0.00 = endocardium, 1.00 = epicardium",
         "-" * 74,
         f"{'depth':>11s} {'sites':>7s} | {'I_Na peak':>10s} {'t_peak':>8s}"
         f" | {'I_Kr peak':>10s} {'t_peak':>8s}",
         f"{'':>11s} {'':>7s} | {'pA/pF':>10s} {'ms':>8s}"
         f" | {'pA/pF':>10s} {'ms':>8s}",
         "-" * 74]
    for b in bands:
        L.append(f"{b.depth_lo:4.2f}-{b.depth_hi:4.2f} {b.n_sites:7d} | "
                 f"{b.ina_peak:10.1f} {b.ina_t_peak:8.1f} | "
                 f"{b.ikr_peak:10.3f} {b.ikr_t_peak:8.1f}")

    def _span(vals):
        return max(vals) - min(vals)

    ina_p = [b.ina_peak for b in bands]
    ina_t = [b.ina_t_peak for b in bands]
    ikr_p = [b.ikr_peak for b in bands]
    ikr_t = [b.ikr_t_peak for b in bands]

    L.append("-" * 74)
    L.append(f"{'span':>11s} {'':>7s} | {_span(ina_p):10.1f} "
             f"{_span(ina_t):8.1f} | {_span(ikr_p):10.3f} {_span(ikr_t):8.1f}")
    L.append("")
    L.append(f"I_Na peak time marches endo -> epi over {_span(ina_t):.1f} ms: "
             f"that is the wavefront crossing the wall.")

    # Classify the repolarisation profile by its *shape*, not by where the
    # argmax happens to land. The two models differ genuinely here: the 1-D
    # strand peaks mid-wall (the M-cell island), while the mesh model uses a
    # monotonic transmural profile and finishes at the endocardial side. Using
    # argmax alone mislabels the mesh, because its two innermost bands are tied
    # to within a few tenths of a millisecond and the winner is then noise.
    span_t = _span(ikr_t)
    tol = max(0.05 * span_t, 1.0)
    diffs = np.diff(ikr_t)
    i_latest = int(np.argmax(ikr_t))
    latest = bands[i_latest]
    where = f"depth {latest.depth_lo:.2f}-{latest.depth_hi:.2f}"

    if np.all(diffs <= tol):
        shape = ("at the endocardial side: repolarisation sweeps epi -> endo "
                 "monotonically across the wall")
    elif np.all(diffs >= -tol):
        shape = ("at the epicardial side: repolarisation sweeps endo -> epi "
                 "monotonically across the wall")
    else:
        shape = (f"mid-wall ({where}), at neither surface, which is the "
                 f"M-cell signature")
    L.append(f"I_Kr peak time spans {span_t:.1f} ms and finishes latest "
             f"{shape}.")
    if _span(ina_p) < 1e-6:
        L.append("I_Na peak amplitude is uniform across the wall by "
                 "construction (every site uses a depth-indexed template, so "
                 "amplitude cannot respond to local coupling).")
    return "\n".join(L)
