"""Matplotlib figures: 12-lead ECG grid, ion-current panel, baseline/drug overlay."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # file output, no GUI required
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .leads import LEAD_NAMES
from .simulate import CellTrace, EcgResult

# clinical 4-column x 3-row layout
_GRID = [["I", "aVR", "V1", "V4"],
         ["II", "aVL", "V2", "V5"],
         ["III", "aVF", "V3", "V6"]]


def _ecg_background(ax, tmax, ymin, ymax):
    """Light ECG-paper gridlines (major 0.2 s / 0.5 mV, minor 0.04 s / 0.1 mV)."""
    ax.set_xticks(np.arange(0, tmax + 1, 200), minor=False)
    ax.set_xticks(np.arange(0, tmax + 1, 40), minor=True)
    ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 0.5, 0.5), minor=False)
    ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 0.1, 0.1), minor=True)
    ax.grid(which="major", color="#f4a8a8", lw=0.6)
    ax.grid(which="minor", color="#fbdede", lw=0.3)
    ax.set_axisbelow(True)


def _currents_at(t, ina_peak, ikr_peak, t_ms: float,
                 half_window: float = 0.6):
    """Wall-wide extreme of I_Na and I_Kr in a short bin around ``t_ms``.

    ``ina_peak`` / ``ikr_peak`` are per-sample spatial extremes, so this works
    for the 1-D strand and the 3-D mesh alike (the mesh cannot supply a full
    field; see :mod:`ecg12sim.mesh_sim`).

    A bin rather than a single sample, for the same reason the text readout in
    :mod:`ecg12sim.timeline` bins: the I_Na wavefront crosses one cell every
    ``dx / CV`` (0.23 ms for the default strand), so reading one instant aliases
    against the cell lattice and under-reports the peak.
    """
    sel = np.abs(t - t_ms) <= half_window
    if not sel.any():
        sel = np.zeros_like(t, dtype=bool)
        sel[int(np.argmin(np.abs(t - t_ms)))] = True
    return float(np.min(ina_peak[sel])), float(np.max(ikr_peak[sel]))


def _lead_peak_time(sig, t, window):
    """Time of this lead's own largest deflection inside ``window``."""
    m = (t >= window[0]) & (t <= window[1])
    if not m.any():
        return float("nan")
    idx = np.flatnonzero(m)
    return float(t[idx[int(np.argmax(np.abs(sig[m])))]])


def _plot_12lead_grid(t, leads, title, qrs_window, t_window,
                      ina_peak=None, ikr_peak=None, caption=None,
                      path=None):
    """Classic 3x4 clinical layout, optionally annotated with both currents.

    When ``ina_peak`` and ``ikr_peak`` are supplied, each panel also reports the
    tracked currents at two instants **chosen per lead**: that lead's own
    largest QRS deflection and its own largest T deflection.

    Worth being precise about what varies per lead and what does not. I_Na and
    I_Kr are properties of the myocardium, so the same two currents underlie all
    twelve traces; there is no such thing as "lead V3's I_Na". What differs is
    *when* each lead peaks, so the currents sampled at each lead's own landmarks
    differ, and that is what these rows report.
    """
    annotate = ina_peak is not None and ikr_peak is not None

    vmax = max(np.max(np.abs(s)) for s in leads.values())
    ymax = vmax * 1.15
    # Reserve a quiet band under the trace for the annotation rather than
    # overprinting the waveform.
    ymin = -vmax * (1.70 if annotate else 1.15)

    fig, axes = plt.subplots(3, 4, figsize=(15, 8.6 if annotate else 7.5),
                             sharex=True, sharey=True)
    for r in range(3):
        for c in range(4):
            name = _GRID[r][c]
            ax = axes[r, c]
            _ecg_background(ax, t[-1], ymin, ymax)
            ax.plot(t, leads[name], color="#111", lw=1.1)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0, t[-1])
            ax.text(0.02, 0.88, name, transform=ax.transAxes,
                    fontsize=11, fontweight="bold")

            if annotate:
                sig = leads[name]
                t_r = _lead_peak_time(sig, t, qrs_window)
                t_t = _lead_peak_time(sig, t, t_window)
                ina_r, ikr_r = _currents_at(t, ina_peak, ikr_peak, t_r)
                ina_t, ikr_t = _currents_at(t, ina_peak, ikr_peak, t_t)
                txt =  (f"R {t_r:5.1f} ms   $I_{{Na}}$ {ina_r:7.1f}   "
                       f"$I_{{Kr}}$ {ikr_r:6.3f}\n"
                       f"T {t_t:5.1f} ms   $I_{{Na}}$ {ina_t:7.1f}   "
                       f"$I_{{Kr}}$ {ikr_t:6.3f}")
                ax.text(0.5, 0.035, txt, transform=ax.transAxes,
                        fontsize=7.0, family="monospace", ha="center",
                        va="bottom", color="#1a1a1a", linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                  edgecolor="#bbbbbb", linewidth=0.6,
                                  alpha=0.96))

            if c == 0:
                ax.set_ylabel("mV")
            if r == 2:
                ax.set_xlabel("time (ms)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if annotate and caption:
        fig.text(0.5, 0.022, caption, ha="center", fontsize=9, style="italic",
                 color="#444444", wrap=True)
        fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


#: Why every panel of the cable figure shows the same numbers.
_CABLE_CAPTION = (
    "Each panel reports the tracked currents (pA/pF) at that lead's own peak "
    "QRS and peak T deflection. All twelve agree here, and that is the "
    "presentation layer showing through rather than a mistake: a 1-D strand has "
    "one electrical axis, so these leads are fixed linear combinations of a "
    "single pseudo-ECG and every one of them peaks at the same instant. Run "
    "--model mesh for leads that are independent functions of a 3-D source, "
    "where the sampled instants differ by up to ~18 ms (QRS) and ~32 ms (T)."
)

#: Caption for the mesh figure, where per-lead sampling is not degenerate.
_MESH_CAPTION = (
    "Each panel reports the tracked currents (pA/pF) at that lead's own peak "
    "QRS and peak T deflection. I_Na and I_Kr are myocardial currents shared by "
    "all twelve leads; what differs per lead is the instant sampled. Because "
    "these leads are genuine potential differences from a 3-D anatomical "
    "source, those instants really do differ, so the values vary panel to "
    "panel. I_Na is large at every R peak and absent at every T peak; I_Kr is "
    "the other way round."
)


def plot_12lead(ecg: EcgResult, title: str, path: str | None = None,
                rep=None, annotate_currents: bool = True):
    """3x4 clinical 12-lead layout for the 1-D cable model."""
    res = ecg.cable
    if rep is not None:
        qrs_window, t_window = rep.qrs_window, rep.t_window
    else:
        from .tracking import ecg_windows
        qrs_window, t_window = ecg_windows(res)

    ina_peak = ikr_peak = None
    if annotate_currents:
        ina_peak = res.I_Na.min(axis=1)
        ikr_peak = res.I_Kr.max(axis=1)

    return _plot_12lead_grid(
        res.t, ecg.leads, title, qrs_window, t_window,
        ina_peak=ina_peak, ikr_peak=ikr_peak,
        caption=_CABLE_CAPTION, path=path)


# def plot_12lead_comparison(base: EcgResult, drug: EcgResult,
#                            ina_block: float = 0.0,
#                            ikr_block: float = 0.0,
#                            path: str | None = None):
#     """Full 12-lead ECG overlay: normal vs channel-blocked beat."""

#     t = base.cable.t
#     td = drug.cable.t

#     vmax = max(
#         max(np.max(np.abs(base.leads[name])) for name in LEAD_NAMES),
#         max(np.max(np.abs(drug.leads[name])) for name in LEAD_NAMES)
#     )
#     limit = max(vmax * 1.15, 0.5)

#     fig, axes = plt.subplots(
#         3, 4, figsize=(15, 8.6), sharex=True, sharey=True
#     )

#     for r in range(3):
#         for c in range(4):
#             name = _GRID[r][c]
#             ax = axes[r, c]

#             _ecg_background(ax, t[-1], -limit, limit)

#             # Normal ECG
#             ax.plot(
#                 t, base.leads[name],
#                 color="#111111", lw=1.3,
#                 label="Normal"
#             )

#             # Drug-blocked ECG
#             ax.plot(
#                 td, drug.leads[name],
#                 color="#d62728", lw=1.3, ls="--",
#                 label="Reduced channel"
#             )

#             ax.set_xlim(0, t[-1])
#             ax.set_ylim(-limit, limit)

#             ax.text(
#                 0.02, 0.88, name,
#                 transform=ax.transAxes,
#                 fontsize=11, fontweight="bold"
#             )

#             if c == 0:
#                 ax.set_ylabel("mV")

#             if r == 2:
#                 ax.set_xlabel("time (ms)")

#     handles, labels = axes[0, 0].get_legend_handles_labels()

#     fig.legend(
#         handles, labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.955),
#         ncol=2,
#         frameon=False,
#         fontsize=10
#     )

#     fig.suptitle(
#         f"12-lead ECG: Normal vs Channel Block "
#         f"(I_Na block {ina_block:.0f}%, "
#         f"I_Kr block {ikr_block:.0f}%)\n"
#         f"Normal QT: {base.qt:.1f} ms | "
#         f"Blocked QT: {drug.qt:.1f} ms",
#         fontsize=14,
#         fontweight="bold",
#         y=1.01
#     )

#     fig.tight_layout(rect=(0, 0, 1, 0.91))

#     if path:
#         fig.savefig(path, dpi=200, facecolor="white",
#                     bbox_inches="tight")
#         plt.close(fig)

#     return fig


def plot_12lead_mesh(beat, leads: dict, title: str, path: str | None = None,
                     annotate_currents: bool = True):
    """3x4 clinical 12-lead layout for the anatomical mesh model.

    ``beat`` is a :class:`ecg12sim.mesh_sim.MeshBeat`, whose per-sample spatial
    extremes are already exact over all nodes, and ``leads`` the output of the
    forward solution.
    """
    ina_peak = beat.ina.peak if annotate_currents else None
    ikr_peak = beat.ikr.peak if annotate_currents else None
    return _plot_12lead_grid(
        beat.t, leads, title, beat.qrs_window, beat.t_window,
        ina_peak=ina_peak, ikr_peak=ikr_peak,
        caption=_MESH_CAPTION, path=path)


# def plot_ion_currents(base: CellTrace, drug: CellTrace,
#                       ina_block: float, ikr_block: float,
#                       path: str | None = None):
#     """Action potential + I_Na and I_Kr traces, baseline vs blocked."""
#     fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
#     tt = base.t

#     axes[0].plot(tt, base.V, color="#1f77b4", lw=1.6, label="baseline")
#     axes[0].plot(drug.t, drug.V, color="#d62728", lw=1.6, ls="--",
#                  label=f"drug (APD90 {drug.apd90:.0f} ms)")
#     axes[0].set_ylabel("Vm (mV)")
#     axes[0].set_title(f"Action potential — {base.cell_type} cell "
#                       f"(baseline APD90 {base.apd90:.0f} ms)")
#     axes[0].legend(loc="upper right")

#     axes[1].plot(tt, base.I_Na, color="#1f77b4", lw=1.4, label="baseline")
#     axes[1].plot(drug.t, drug.I_Na, color="#d62728", lw=1.4, ls="--",
#                  label=f"I_Na block {ina_block:.0f}%")
#     axes[1].set_ylabel("I_Na (pA/pF)")
#     axes[1].set_title("Fast sodium current I_Na (phase-0 upstroke)")
#     axes[1].legend(loc="lower right")

#     axes[2].plot(tt, base.I_Kr, color="#1f77b4", lw=1.4, label="baseline")
#     axes[2].plot(drug.t, drug.I_Kr, color="#d62728", lw=1.4, ls="--",
#                  label=f"I_Kr block {ikr_block:.0f}%")
#     axes[2].set_ylabel("I_Kr (pA/pF)")
#     axes[2].set_title("Rapid delayed-rectifier K+ current I_Kr (repolarisation)")
#     axes[2].set_xlabel("time (ms)")
#     axes[2].legend(loc="upper right")

#     for ax in axes:
#         ax.grid(color="#dddddd", lw=0.5)
#         ax.set_axisbelow(True)
#     fig.tight_layout()
#     if path:
#         fig.savefig(path, dpi=130)
#         plt.close(fig)
#     return fig


# def plot_ecg_comparison(base: EcgResult, drug: EcgResult,
#                         leads_to_show=("II", "V2", "V5"),
#                         path: str | None = None):
#     """Overlay baseline vs drug for a few leads to highlight QT / QRS changes."""
#     t = base.cable.t
#     fig, axes = plt.subplots(len(leads_to_show), 1,
#                              figsize=(10, 2.4 * len(leads_to_show)), sharex=True)
#     if len(leads_to_show) == 1:
#         axes = [axes]
#     for ax, name in zip(axes, leads_to_show):
#         ax.plot(t, base.leads[name], color="#1f77b4", lw=1.3, label="baseline")
#         ax.plot(drug.cable.t, drug.leads[name], color="#d62728", lw=1.3, ls="--",
#                 label="drug")
#         ax.set_ylabel(f"{name} (mV)")
#         ax.grid(color="#dddddd", lw=0.5)
#         ax.set_axisbelow(True)
#         ax.legend(loc="upper right", fontsize=8)
#     axes[-1].set_xlabel("time (ms)")
#     fig.suptitle(f"Baseline QT {base.qt:.0f} ms  vs  drug QT {drug.qt:.0f} ms",
#                  fontsize=13, fontweight="bold")
#     fig.tight_layout(rect=(0, 0, 1, 0.96))
#     if path:
#         fig.savefig(path, dpi=130)
#         plt.close(fig)
#     return fig

def plot_12lead_overlay(base: EcgResult, drug: EcgResult, title: str,
                        ina_block: float = 0.0, ikr_block: float = 0.0,
                        path: str | None = None):
    """Full 3x4 12-lead overlay with per-condition R/T and current readouts."""
    from .tracking import ecg_windows

    b, d = base.cable, drug.cable
    fig, axes = plt.subplots(3, 4, figsize=(16, 10),
                             sharex=True, sharey=True)

    for r in range(3):
        for c in range(4):
            name = _GRID[r][c]
            ax = axes[r, c]
            sig_b, sig_d = base.leads[name], drug.leads[name]

            vmax = max(float(np.max(np.abs(sig_b))),
                       float(np.max(np.abs(sig_d))), 0.1)
            ymin, ymax = -1.75 * vmax, 1.15 * vmax
            _ecg_background(ax, b.t[-1], ymin, ymax)

            ax.plot(b.t, sig_b, color="#111111", lw=1.15,
                    label="Normal")
            ax.plot(d.t, sig_d, color="#d62728", lw=1.15,
                    ls="--", label="Blocked")

            ax.set_xlim(0, b.t[-1])
            ax.set_ylim(ymin, ymax)
            ax.text(0.02, 0.91, name, transform=ax.transAxes,
                    fontsize=10, fontweight="bold")

            qb, tb = ecg_windows(b)
            qd, td = ecg_windows(d)

            trb = _lead_peak_time(sig_b, b.t, qb)
            ttb = _lead_peak_time(sig_b, b.t, tb)
            trd = _lead_peak_time(sig_d, d.t, qd)
            ttd = _lead_peak_time(sig_d, d.t, td)

            inab_r, ikrb_r = _currents_at(
                b.t, b.I_Na.min(axis=1), b.I_Kr.max(axis=1), trb)
            inab_t, ikrb_t = _currents_at(
                b.t, b.I_Na.min(axis=1), b.I_Kr.max(axis=1), ttb)
            inad_r, ikrd_r = _currents_at(
                d.t, d.I_Na.min(axis=1), d.I_Kr.max(axis=1), trd)
            inad_t, ikrd_t = _currents_at(
                d.t, d.I_Na.min(axis=1), d.I_Kr.max(axis=1), ttd)

            txt_base = (
            f"R {trb:5.1f} ms  $I_{{Na}}$ {inab_r:7.1f}  "
            f"$I_{{Kr}}$ {ikrb_r:6.3f}\n"
            f"T {ttb:5.1f} ms  $I_{{Na}}$ {inab_t:7.1f}  "
            f"$I_{{Kr}}$ {ikrb_t:6.3f}"
            )

            txt_drug = (
            f"R {trd:5.1f} ms  $I_{{Na}}$ {inad_r:7.1f}  "
            f"$I_{{Kr}}$ {ikrd_r:6.3f}\n"
            f"T {ttd:5.1f} ms  $I_{{Na}}$ {inad_t:7.1f}  "
            f"$I_{{Kr}}$ {ikrd_t:6.3f}"
            )

            ax.text(
            0.02, 0.035, txt_base,
            transform=ax.transAxes,
            fontsize=5.8,
            family="monospace",
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.94
            )
            )

            ax.text(
            0.98, 0.035, txt_drug,
            transform=ax.transAxes,
            fontsize=5.8,
            family="monospace",
            ha="right",
            va="bottom",
            color="red",
            bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="red",
            linewidth=0.5,
            alpha=0.94
            )
            )


            if c == 0:
                ax.set_ylabel("mV")
            if r == 2:
                ax.set_xlabel("time (ms)")

    handles = [
        Line2D([0], [0], color="#111", lw=1.5, label="Normal"),
        Line2D([0], [0], color="#d62728", lw=1.5,
               ls="--", label="Channel block")
    ]
    fig.legend(handles=handles, loc="upper center",bbox_to_anchor = (0.5, 0.955),
               ncol=2, frameon=False, fontsize = 10)

    fig.suptitle(
        f"{title}\n" 
        f"I_Na block {ina_block:.0f}% | "
        f"I_Kr block {ikr_block:.0f}%\n"
        f"QT: {base.qt:.1f} ms (normal) vs "
        f"{drug.qt:.1f} ms (blocked)",
        fontsize=13, fontweight="bold", y=.995
    )

    fig.tight_layout(rect=(0, 0, 1, .89))

    if path:
        fig.savefig(path, dpi=170, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)

    return fig

# ---------------------------------------------------------------------------
# Tracking views: observe I_Na and I_Kr across the wall during a normal beat
# ---------------------------------------------------------------------------

def plot_current_maps(ecg: EcgResult, rep, path: str | None = None):
    """Space-time maps of I_Na and I_Kr aligned under the lead-II ECG.

    Rows share the time axis so you can read off, at a glance, that the I_Na
    wavefront coincides with the QRS and the dispersed I_Kr peak writes the T wave.
    The y-axis of each map is transmural position (endocardium at the bottom,
    epicardium at the top).
    """
    res = ecg.cable
    t, x = res.t, res.x
    xmm = x * 10.0
    extent = (t[0], t[-1], xmm[0], xmm[-1])            # position in mm

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5),
                             gridspec_kw={"height_ratios": [1.0, 1.4, 1.4]})

    # --- lead II ECG with the two phase windows shaded (full beat) ---
    ax = axes[0]
    ax.plot(t, ecg.leads["II"], color="#111", lw=1.4)
    ax.axvspan(*rep.qrs_window, color="#1f77b4", alpha=0.18, label="QRS (I_Na active)")
    ax.axvspan(*rep.t_window, color="#d62728", alpha=0.15, label="T wave (I_Kr peak)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("lead II (mV)")
    ax.set_title("ECG produced by the beat, with depolarisation / repolarisation windows")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)

    # --- I_Kr map, on the same full time axis as the ECG above ---
    ax = axes[1]
    im2 = ax.imshow(res.I_Kr.T, aspect="auto", origin="lower", extent=extent,
                    cmap="hot_r", vmin=0.0, vmax=res.I_Kr.max())
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("transmural position (mm)\nendo -> epi")
    ax.set_title("I_Kr  —  sustained; its dispersed peak (dark band) writes the T wave above")
    ax.set_xlabel("time (ms)")
    fig.colorbar(im2, ax=ax, label="I_Kr (pA/pF)", pad=0.01)

    # --- I_Na map, zoomed: it is active for only a few ms of the whole beat ---
    ax = axes[2]
    im1 = ax.imshow(res.I_Na.T, aspect="auto", origin="lower", extent=extent,
                    cmap="Blues_r", vmin=res.I_Na.min(), vmax=0.0)
    z0, z1 = rep.qrs_window[0] - 2.0, rep.qrs_window[1] + 4.0
    ax.set_xlim(z0, z1)
    # overlay the wavefront trajectory (activation time vs depth)
    ax.plot(res.activation_time, xmm, color="#2ca02c", lw=1.6, ls="--",
            label=f"wavefront, {rep.conduction_velocity:.0f} cm/s")
    ax.set_ylabel("transmural position (mm)\nendo -> epi")
    ax.set_title(f"I_Na  —  brief wavefront, active only "
                 f"{rep.ina_active_fraction * 100:.1f}% of the beat "
                 f"(time axis zoomed to {z0:.0f}-{z1:.0f} ms)")
    ax.set_xlabel("time (ms)  [zoomed]")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im1, ax=ax, label="I_Na (pA/pF)", pad=0.01)

    fig.suptitle("Tracking I_Na and I_Kr across the ventricular wall (no block applied)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_transmural_traces(ecg: EcgResult, rep, path: str | None = None):
    """I_Na and I_Kr time-courses sampled at endo / mid / epi sites.

    Shows the natural transmural variation: the wavefront reaches each site at a
    different time (I_Na), and each site repolarises at a different time with a
    different I_Kr magnitude (M cells latest and largest).
    """
    res = ecg.cable
    t = res.t
    n = len(res.x)
    sites = [("endo", int(0.12 * n), "#1f77b4"),
             ("mid (M)", int(0.45 * n), "#2ca02c"),
             ("epi", int(0.88 * n), "#d62728")]

    # NOTE: axes are deliberately NOT shared. Panel 3 is a zoom on the activation
    # window; with a shared x-axis its xlim would collapse every other panel too.
    fig, axes = plt.subplots(4, 1, figsize=(11, 11.5))
    full = (t[0], t[-1])
    zoom = (rep.qrs_window[0] - 3.0, rep.qrs_window[1] + 5.0)

    ax = axes[0]
    for label, i, c in sites:
        ax.plot(t, res.Vm[:, i], color=c, lw=1.5, label=f"{label} (x={res.x[i]*10:.1f} mm)")
    ax.set_xlim(*full)
    ax.set_ylabel("Vm (mV)")
    ax.set_title("Action potentials at three transmural depths "
                 "(M cell repolarises last)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for label, i, c in sites:
        ax.plot(t, res.I_Kr[:, i], color=c, lw=1.5, label=label)
    ax.axvspan(*rep.t_window, color="#d62728", alpha=0.12, label="T-wave window")
    ax.set_xlim(*full)
    ax.set_ylabel("I_Kr (pA/pF)")
    ax.set_title("I_Kr — small and sustained; peaks as each site repolarises")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    for label, i, c in sites:
        ax.plot(t, res.I_Na[:, i], color=c, lw=1.5, label=label)
    ax.set_xlim(*full)
    ax.set_ylabel("I_Na (pA/pF)")
    ax.set_title("I_Na on the same full-beat axis — three spikes so brief they are "
                 "nearly invisible")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[3]
    for label, i, c in sites:
        ax.plot(t, res.I_Na[:, i], color=c, lw=1.8, label=label)
    ax.set_xlim(*zoom)
    ax.set_ylabel("I_Na (pA/pF)")
    ax.set_title(f"I_Na zoomed to {zoom[0]:.0f}-{zoom[1]:.0f} ms — the stagger between "
                 f"sites is conduction ({rep.conduction_velocity:.0f} cm/s)")
    ax.set_xlabel("time (ms)  [zoomed]")
    ax.legend(loc="lower right", fontsize=8)

    for a in axes:
        a.grid(color="#dddddd", lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("Transmural variation of I_Na and I_Kr during one normal beat",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_current_ecg_correlation(ecg: EcgResult, rep, path: str | None = None):
    """Wall-summed I_Na and I_Kr against the ECG they generate.

    Summing each current over the whole wall gives the tissue-level current that
    the ECG actually reflects, making the QRS<->I_Na and T<->I_Kr correspondence
    explicit on a single time axis.
    """
    res = ecg.cable
    t = res.t

    # Wall-summed activity. I_Na is summed as a magnitude because it is inward,
    # and lightly smoothed: with a finite number of discrete cells the raw sum of
    # 1 ms spikes firing at staggered times is lumpy, which obscures the envelope.
    ina_act = np.abs(res.I_Na).sum(axis=1)
    k = 7
    ina_act = np.convolve(ina_act, np.ones(k) / k, mode="same")
    ikr_act = res.I_Kr.sum(axis=1)

    ina_n = ina_act / (ina_act.max() or 1.0)
    ikr_n = ikr_act / (ikr_act.max() or 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.1]})

    ax = axes[0]
    ax.plot(t, ecg.leads["II"], color="#111", lw=1.6)
    ax.axvspan(*rep.qrs_window, color="#1f77b4", alpha=0.15)
    ax.axvspan(*rep.t_window, color="#d62728", alpha=0.13)
    ax.set_ylabel("lead II (mV)")
    ax.set_title("The ECG this beat produces")
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.fill_between(t, 0, ina_n, color="#1f77b4", alpha=0.35)
    ax.plot(t, ina_n, color="#1f77b4", lw=1.5, label="|I_Na| summed over the wall")
    ax.fill_between(t, 0, ikr_n, color="#d62728", alpha=0.30)
    ax.plot(t, ikr_n, color="#d62728", lw=1.5, label="I_Kr summed over the wall")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("wall-summed current\n(each normalised to its own peak)")
    ax.set_xlabel("time (ms)")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)

    fig.suptitle("I_Na drives the QRS, I_Kr drives the T wave",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_combined_waveforms(ecg: EcgResult, rep, cell_index: int | None = None,
                            path: str | None = None):
    """One image: the ECG with its tracked I_Na and I_Kr, on a shared time axis.

    All three panels share a single x-axis, so features line up exactly. Two
    vertical guide lines (at the I_Na peak and the I_Kr peak) and the shaded QRS
    and T windows are repeated in every panel so the alignment is verifiable by
    eye rather than asserted.

    The ECG is generated by the whole ventricular wall; the two current traces are
    recorded from one representative mid-wall myocyte, which is why the ECG
    deflections are broader than the single-cell currents that build them.
    """
    res = ecg.cable
    t = res.t
    n = len(res.x)
    i = int(0.45 * n) if cell_index is None else cell_index
    depth_mm = res.x[i] * 10.0

    ina = res.I_Na[:, i]
    ikr = res.I_Kr[:, i]
    ecg_ii = ecg.leads["II"]

    t_ina = float(t[np.argmin(ina)])
    t_ikr = float(t[np.argmax(ikr)])
    ina_pk = float(ina.min())
    ikr_pk = float(ikr.max())

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)

    def decorate(ax):
        """Phase windows + alignment guides, identical on every panel."""
        ax.axvspan(*rep.qrs_window, color="#1f77b4", alpha=0.13, zorder=0)
        ax.axvspan(*rep.t_window, color="#d62728", alpha=0.11, zorder=0)
        ax.axvline(t_ina, color="#1f77b4", lw=1.1, ls=":", zorder=1)
        ax.axvline(t_ikr, color="#d62728", lw=1.1, ls=":", zorder=1)
        ax.grid(color="#e4e4e4", lw=0.5)
        ax.set_axisbelow(True)

    # ------------------------------------------------ panel 1: the ECG
    ax = axes[0]
    decorate(ax)
    ax.plot(t, ecg_ii, color="#111111", lw=1.8, zorder=3)
    ax.set_ylabel("ECG, lead II\n(mV)", fontsize=11, fontweight="bold")
    ax.set_title("A.  12-lead ECG (lead II) generated by the simulated beat",
                 loc="left", fontsize=12, fontweight="bold")

    r_idx = int(np.argmax(ecg_ii))
    ax.annotate("QRS complex\n(depolarisation)",
                xy=(t[r_idx], ecg_ii[r_idx]),
                xytext=(70, 0.95), fontsize=10, color="#1f77b4",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    tmask = t > rep.qrs_window[1] + 30.0
    tpk_idx = int(np.argmax(np.where(tmask, ecg_ii, -np.inf)))
    ax.annotate("T wave\n(repolarisation)",
                xy=(t[tpk_idx], ecg_ii[tpk_idx]),
                xytext=(t[tpk_idx] + 55, 0.75), fontsize=10, color="#d62728",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
    ax.set_ylim(min(ecg_ii.min() * 1.35, -0.35), max(ecg_ii.max() * 1.35, 1.45))

    # ------------------------------------------------ panel 2: I_Na
    ax = axes[1]
    decorate(ax)
    ax.fill_between(t, 0.0, ina, color="#1f77b4", alpha=0.30, zorder=2)
    ax.plot(t, ina, color="#1f77b4", lw=1.8, zorder=3)
    ax.set_ylabel("$I_{Na}$\n(pA/pF)", fontsize=11, fontweight="bold")
    ax.set_title("B.  Tracked $I_{Na}$ — fast inward sodium current "
                 "(brief, large; drives the QRS)",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylim(ina_pk * 1.22, abs(ina_pk) * 0.16)
    ax.annotate(f"peak {ina_pk:.0f} pA/pF at t = {t_ina:.1f} ms\n"
                f"active only {rep.ina_active_fraction * 100:.1f}% of the beat",
                xy=(t_ina, ina_pk), xycoords="data",
                xytext=(0.17, 0.30), textcoords="axes fraction",
                fontsize=10, color="#1f77b4", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))

    # Inset zoom: the spike is ~1 ms wide, so its shape is invisible on a 500 ms
    # axis. The inset has its own axes, so the shared alignment above is untouched.
    # Placed over the electrically quiet plateau so it cannot be mistaken for data
    # belonging to the shaded T-wave window.
    axin = ax.inset_axes([0.42, 0.12, 0.33, 0.52])
    z0, z1 = t_ina - 3.0, t_ina + 5.0
    zsel = (t >= z0) & (t <= z1)
    axin.fill_between(t[zsel], 0.0, ina[zsel], color="#1f77b4", alpha=0.30)
    axin.plot(t[zsel], ina[zsel], color="#1f77b4", lw=1.8)
    axin.set_xlim(z0, z1)
    axin.set_title(f"inset — same spike, time axis zoomed to {z0:.0f}-{z1:.0f} ms",
                   fontsize=8.5, fontweight="bold")
    axin.tick_params(labelsize=7.5)
    axin.grid(color="#e4e4e4", lw=0.4)
    axin.set_facecolor("white")
    for s in axin.spines.values():
        s.set_edgecolor("#1f77b4")
        s.set_linewidth(1.3)

    # ------------------------------------------------ panel 3: I_Kr
    ax = axes[2]
    decorate(ax)
    ax.fill_between(t, 0.0, ikr, color="#d62728", alpha=0.28, zorder=2)
    ax.plot(t, ikr, color="#d62728", lw=1.8, zorder=3)
    ax.set_ylabel("$I_{Kr}$\n(pA/pF)", fontsize=11, fontweight="bold")
    ax.set_title("C.  Tracked $I_{Kr}$ — rapid delayed-rectifier potassium current "
                 "(small, sustained; drives the T wave)",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("time (ms)  —  shared by all three panels", fontsize=11)
    ax.set_ylim(-ikr_pk * 0.12, ikr_pk * 1.42)
    ax.annotate(f"peak {ikr_pk:.3f} pA/pF at t = {t_ikr:.1f} ms\n"
                f"active {rep.ikr_active_fraction * 100:.1f}% of the beat\n"
                f"(this mid-wall M cell repolarises last, so its peak falls at\n"
                f"the end of the T wave; the wave itself is written by the\n"
                f"spread of repolarisation across the whole wall)",
                xy=(t_ikr, ikr_pk), xycoords="data",
                xytext=(0.30, 0.60), textcoords="axes fraction",
                fontsize=9.5, color="#d62728", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
    ax.set_xlim(t[0], t[-1])

    # ------------------------------------------------ shared legend / caption
    handles = [
        Patch(facecolor="#1f77b4", alpha=0.13,
              label=f"QRS / depolarisation window "
                    f"({rep.qrs_window[0]:.0f}-{rep.qrs_window[1]:.0f} ms)"),
        Patch(facecolor="#d62728", alpha=0.11,
              label=f"T wave / repolarisation window "
                    f"({rep.t_window[0]:.0f}-{rep.t_window[1]:.0f} ms)"),
        Line2D([0], [0], color="#1f77b4", ls=":", lw=1.4,
               label=f"$I_{{Na}}$ peak, t = {t_ina:.1f} ms"),
        Line2D([0], [0], color="#d62728", ls=":", lw=1.4,
               label=f"$I_{{Kr}}$ peak, t = {t_ikr:.1f} ms"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5,
               frameon=True, bbox_to_anchor=(0.5, 0.012))

    fig.suptitle("ECG with its tracked $I_{Na}$ and $I_{Kr}$ currents — "
                 "one normal beat, shared time axis",
                 fontsize=14.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.055,
             f"ECG is generated by the whole ventricular wall; currents are "
             f"recorded from a mid-wall myocyte at {depth_mm:.1f} mm "
             f"(QT = {ecg.qt:.0f} ms, conduction velocity = "
             f"{rep.conduction_velocity:.0f} cm/s). No channel block applied.",
             ha="center", fontsize=9.5, style="italic", color="#444444")

    fig.tight_layout(rect=(0, 0.085, 1, 0.97))
    if path:
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Single-cell action potential and conductance sweeps
# ---------------------------------------------------------------------------

#: Currents grouped by role and, just as importantly, by magnitude.
#:
#: I_to is kept with the early currents rather than the repolarising ones. That
#: is both physiologically right (it is the transient outward current, peaking
#: within a few ms) and practically necessary: its peak of ~14 pA/pF is roughly
#: 17x I_Kr's, so sharing an axis would flatten I_Kr - one of the two currents
#: this project tracks - into an invisible line.
_CURRENT_GROUPS = (
    ("Early currents: I_CaL inward, I_to outward",
     [("I_CaL", "#ff7f0e"), ("I_to", "#17becf")]),
    ("Repolarising K currents (note the scale: all under 3 pA/pF)",
     [("I_Kr", "#d62728"), ("I_Ks", "#9467bd"), ("I_K1", "#2ca02c")]),
    ("Background and pump",
     [("I_pK", "#7f7f7f"), ("I_NaK", "#bcbd22"), ("I_bNa", "#8c564b")]),
)


def plot_complete_ap(t, V, traces, met, cell_type: str,
                     path: str | None = None):
    """The complete action potential: Vm plus every current in the model.

    Five stacked panels on one shared time axis. I_Na gets a panel of its own
    with an inset, because it is two orders of magnitude larger than the
    repolarising currents and about a thousand times briefer, so plotting it
    alongside them would flatten everything else to a line.
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios":
                                          [1.5, 1.1, 1.0, 1.0, 1.0]})

    # ---- Vm ----
    ax = axes[0]
    ax.plot(t, V, color="#111", lw=2.0)
    ax.axhline(met.v_rest, color="#888", lw=0.8, ls=":")
    ax.set_ylabel("Vm (mV)", fontsize=11, fontweight="bold")
    ax.set_title(f"A.  Action potential — {cell_type} ventricular myocyte",
                 loc="left", fontsize=12, fontweight="bold")
    # APD markers
    i_pk = int(np.argmax(V))
    for frac, lab, col in ((0.50, "APD50", "#1f77b4"), (0.90, "APD90", "#d62728")):
        v_x = met.v_peak - frac * (met.v_peak - met.v_rest)
        below = np.flatnonzero(V[i_pk:] < v_x)
        if below.size:
            tx = t[i_pk + below[0]]
            ax.axhline(v_x, color=col, lw=0.8, ls="--", alpha=0.7)
            ax.annotate(f"{lab} = {met.apd50 if frac == 0.5 else met.apd90:.0f} ms",
                        xy=(tx, v_x), xytext=(tx + 18, v_x + 9),
                        fontsize=9.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.text(0.985, 0.90,
            f"rest {met.v_rest:.1f} mV\npeak {met.v_peak:+.1f} mV\n"
            f"dV/dt max {met.dvdt_max:.0f} mV/ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbb"))

    # ---- I_Na, on its own scale ----
    ax = axes[1]
    ina = traces["I_Na"]
    ax.fill_between(t, 0, ina, color="#1f77b4", alpha=0.30)
    ax.plot(t, ina, color="#1f77b4", lw=1.6)
    ax.set_ylabel("$I_{Na}$\n(pA/pF)", fontsize=11, fontweight="bold")
    ax.set_title(f"B.  $I_{{Na}}$ — peak {ina.min():.1f} pA/pF, "
                 f"two orders of magnitude larger than any other current "
                 f"and over in a few ms",
                 loc="left", fontsize=11, fontweight="bold")
    axin = ax.inset_axes([0.55, 0.14, 0.40, 0.62])
    zs = (t >= 0) & (t <= 6)
    axin.fill_between(t[zs], 0, ina[zs], color="#1f77b4", alpha=0.30)
    axin.plot(t[zs], ina[zs], color="#1f77b4", lw=1.6)
    axin.set_title("inset — first 6 ms", fontsize=8.5, fontweight="bold")
    axin.tick_params(labelsize=7.5)
    axin.grid(color="#eee", lw=0.4)
    for s in axin.spines.values():
        s.set_edgecolor("#1f77b4")

    # ---- grouped currents ----
    for ax, (title, members) in zip(axes[2:], _CURRENT_GROUPS):
        for name, col in members:
            tr = traces[name]
            ax.plot(t, tr, color=col, lw=1.5,
                    label=f"{name}  peak {tr[int(np.argmax(np.abs(tr)))]:+.2f}")
        ax.set_ylabel("pA/pF", fontsize=10, fontweight="bold")
        ax.set_title(f"{title}", loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, ncol=2)

    axes[-1].set_xlabel("time (ms)  —  shared by all panels", fontsize=11)
    for a in axes:
        a.grid(color="#e8e8e8", lw=0.5)
        a.set_axisbelow(True)
        a.set_xlim(t[0], t[-1])

    fig.suptitle("Complete myocardial cell action potential and every "
                 "underlying current",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    if path:
        fig.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
    return fig

def plot_complete_ap_comparison(
    t_base, v_base, tr_base, met_base,
    t_drug, v_drug, tr_drug, met_drug,
    cell_type: str, label: str = "drug condition",
    path: str | None = None
):
    """Overlay normal and drug APs plus all nine ionic-current traces."""
    fig, axes = plt.subplots(
        5, 1, figsize=(12, 13), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.1, 1, 1, 1]}
    )

    normal_color = "#111111"
    drug_color = "#d62728"

    # Panel A: membrane voltage
    ax = axes[0]
    ax.plot(t_base, v_base, color=normal_color, lw=1.8,
            label="Normal")
    ax.plot(t_drug, v_drug, color=drug_color, lw=1.6,
            ls="--", label=label)
    ax.set_ylabel("Vm (mV)")
    ax.set_title(
        f"A. Membrane voltage — {cell_type} ventricular myocyte",
        loc="left"
    )
    ax.legend(loc="best", fontsize=8)
    ax.text(
        .99, .96,
        f"APD90: {met_base.apd90:.1f} ms → "
        f"{met_drug.apd90:.1f} ms",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9,
        bbox=dict(fc="white", ec="#bbb", alpha=.9)
    )

    # Panel B: fast sodium current
    ax = axes[1]
    for traces, times, color, style, name in (
        (tr_base, t_base, normal_color, "-", "Normal"),
        (tr_drug, t_drug, drug_color, "--", label)
    ):
        ax.plot(times, traces["I_Na"], color=color,
                ls=style, lw=1.4, label=name)

    ax.set_ylabel("I_Na\n(pA/pF)")
    ax.set_title("B. Fast sodium current", loc="left")
    ax.legend(fontsize=8)

    # Panels C-E: all remaining ionic currents
    for ax, (heading, members) in zip(axes[2:], _CURRENT_GROUPS):
        for name, color in members:
            ax.plot(
                t_base, tr_base[name], color=color,
                lw=1.5, label=f"{name} normal"
            )
            ax.plot(
                t_drug, tr_drug[name], color=color,
                lw=1.3, ls="--", label=f"{name} blocked"
            )

        ax.set_ylabel("pA/pF")
        ax.set_title(heading, loc="left")
        ax.legend(fontsize=7, ncol=3)

    axes[-1].set_xlabel("time (ms)")

    for ax in axes:
        ax.grid(color="#e8e8e8", lw=.5)
        ax.set_axisbelow(True)
        ax.set_xlim(t_base[0], t_base[-1])

    fig.suptitle(
        "Complete myocardial action potential: "
        "normal vs drug condition",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, .975))

    if path:
        fig.savefig(path, dpi=160, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)

    return fig
def _sweep_colors(scales):
    """Diverging colours about baseline: blue below 1.0, red above."""
    cmap = plt.get_cmap("coolwarm")
    lg = np.log2(np.asarray(scales, dtype=float))
    span = max(abs(lg).max(), 1e-9)
    return [cmap(0.5 + 0.5 * v / span) for v in lg]


def plot_ap_sweeps(ina_res, ikr_res, path: str | None = None):
    """How the action potential responds to scaling I_Na, then I_Kr.

    Laid out so the dissociation is readable directly off the figure: I_Na
    reshapes the upstroke and barely touches APD, while I_Kr reshapes
    repolarisation and leaves the upstroke identical. The bottom row makes that
    quantitative, plotting both metrics against scale for both sweeps.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    for row, res in enumerate((ina_res, ikr_res)):
        cols = _sweep_colors(res.scales)
        b = res.baseline_index
        sym = "$I_{Na}$" if res.which == "I_Na" else "$I_{Kr}$"

        # left: the action potential family
        ax = axes[row, 0]
        for i, s in enumerate(res.scales):
            ax.plot(res.t, res.Vm[i], color=cols[i], lw=1.3,
                    zorder=2, alpha=0.95)
        ax.plot(res.t, res.Vm[b], color="#111", lw=2.2, zorder=3,
                label="baseline (1.00x)")
        ax.set_xlim(res.t[0], res.t[-1])
        ax.set_ylabel("Vm (mV)", fontsize=11, fontweight="bold")
        ax.set_xlabel("time (ms)")
        ax.set_title(f"{'AC'[row]}.  {sym} scaled "
                     f"{res.scales.min():.2f}x to {res.scales.max():.2f}x "
                     f"({len(res.scales)} levels)",
                     loc="left", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)

        # Say so when the traces coincide, rather than leaving a panel that
        # looks like a rendering failure. For I_Na they very nearly do.
        ok = [m for m in res.metrics if m.excited]
        apd_span = max(m.apd90 for m in ok) - min(m.apd90 for m in ok)
        if apd_span < 10.0:
            ax.text(0.03, 0.06,
                    f"all {len(res.scales)} traces superimposed: {sym} barely "
                    f"affects the plateau or\nrepolarisation "
                    f"(APD90 spans only {apd_span:.1f} ms across the whole "
                    f"sweep).\nIts effect is on the upstroke — see panel B.",
                    transform=ax.transAxes, fontsize=9, color="#333",
                    style="italic", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                              edgecolor="#d9c97a"))

        # right: the feature that current actually controls
        ax = axes[row, 1]
        if res.which == "I_Na":
            for i in range(len(res.scales)):
                ax.plot(res.t, res.Vm[i], color=cols[i], lw=1.5)
            ax.plot(res.t, res.Vm[b], color="#111", lw=2.2)
            ax.set_xlim(0, 4.0)
            ax.set_ylabel("Vm (mV)", fontsize=11, fontweight="bold")
            ax.set_title("B.  the same traces, upstroke zoomed to 0-4 ms — "
                         "this is what $I_{Na}$ controls",
                         loc="left", fontsize=11, fontweight="bold")
        else:
            for i in range(len(res.scales)):
                ax.plot(res.t, res.I_Kr[i], color=cols[i], lw=1.5)
            ax.plot(res.t, res.I_Kr[b], color="#111", lw=2.2)
            ax.set_xlim(res.t[0], res.t[-1])
            ax.set_ylabel("$I_{Kr}$ (pA/pF)", fontsize=11, fontweight="bold")
            ax.set_title("D.  the $I_{Kr}$ traces themselves — larger and "
                         "earlier as the conductance rises",
                         loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("time (ms)")

    # ---- bottom row: the numbers ----
    styles = {"I_Na": ("#1f77b4", "o", "$I_{Na}$ scaled"),
              "I_Kr": ("#d62728", "s", "$I_{Kr}$ scaled")}

    ax = axes[2, 0]
    for res in (ina_res, ikr_res):
        col, mk, lab = styles[res.which]
        ok = [m for m in res.metrics if m.excited]
        ax.plot([m.scale for m in ok], [m.dvdt_max for m in ok],
                color=col, marker=mk, lw=1.8, ms=5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("conductance scale (log2)")
    ax.set_ylabel("dV/dt max (mV/ms)", fontsize=11, fontweight="bold")
    ax.set_title("E.  upstroke velocity responds to $I_{Na}$ only",
                 loc="left", fontsize=11, fontweight="bold")
    ax.axvline(1.0, color="#888", lw=0.9, ls=":")
    ax.legend(fontsize=9)

    ax = axes[2, 1]
    for res in (ina_res, ikr_res):
        col, mk, lab = styles[res.which]
        ok = [m for m in res.metrics if m.excited]
        ax.plot([m.scale for m in ok], [m.apd90 for m in ok],
                color=col, marker=mk, lw=1.8, ms=5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("conductance scale (log2)")
    ax.set_ylabel("APD90 (ms)", fontsize=11, fontweight="bold")
    ax.set_title("F.  action-potential duration responds to $I_{Kr}$",
                 loc="left", fontsize=11, fontweight="bold")
    ax.axvline(1.0, color="#888", lw=0.9, ls=":")
    ax.legend(fontsize=9)

    for a in axes.ravel():
        a.grid(color="#e8e8e8", lw=0.5)
        a.set_axisbelow(True)

    ina_ok = [m for m in ina_res.metrics if m.excited]
    ikr_ok = [m for m in ikr_res.metrics if m.excited]
    caption = (
        f"Blue traces are below baseline, red above; black is baseline. "
        f"Scaling $I_{{Na}}$ moves dV/dt max over "
        f"{max(m.dvdt_max for m in ina_ok) - min(m.dvdt_max for m in ina_ok):.0f} "
        f"mV/ms while APD90 shifts only "
        f"{max(m.apd90 for m in ina_ok) - min(m.apd90 for m in ina_ok):.1f} ms. "
        f"Scaling $I_{{Kr}}$ leaves dV/dt max completely unchanged and moves "
        f"APD90 over "
        f"{max(m.apd90 for m in ikr_ok) - min(m.apd90 for m in ikr_ok):.1f} ms. "
        f"Reduced $I_{{Kr}}$ prolonging the action potential is the mechanism "
        f"behind drug-induced QT prolongation."
    )
    fig.suptitle("How the action potential changes when $I_{Na}$ and "
                 "$I_{Kr}$ are scaled separately",
                 fontsize=14.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, caption, ha="center", fontsize=9.5, style="italic",
             color="#444", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    if path:
        fig.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
    return fig
