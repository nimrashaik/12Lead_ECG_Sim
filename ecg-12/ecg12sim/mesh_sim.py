"""Reaction-eikonal assembly of a beat on the anatomical mesh.

Combines the two halves: :mod:`ecg12sim.eikonal` gives each node an activation
time, :mod:`ecg12sim.templates` gives each transmural depth an action potential
and its I_Na / I_Kr traces, and this module puts them together as

    Vm(node, t) = template[depth(node)](t - activation(node))

Why the spatial field is never stored
-------------------------------------

Materialising ``Vm`` for every node at every sample would cost
``355,095 nodes x ~6,000 samples x 8 bytes`` which is about 17 GB per field, and
three fields are needed. So the whole beat is assembled in *reduced* form
instead, and the reductions are computed exactly rather than by sampling.

The trick is that every node's trace is the same template shifted in time. For
any spatially-summed quantity, the sum over nodes of a shifted template is a
**convolution of that template with the histogram of activation times**:

    sum_over_nodes template(t - AT_node) = (template * activation_histogram)(t)

Because activation spans only ~80 ms, each histogram has a few hundred non-zero
bins, so the convolution is a short weighted accumulation rather than an FFT,
and the result is exact to the activation-time binning. Spatial *maxima* follow
the same structure with ``maximum`` in place of ``+`` (a grayscale dilation over
the set of activation times present), and "fraction of nodes active" is the same
convolution applied to the template's threshold indicator.

So the peak, mean and active-fraction columns of the readout are exact over all
355k nodes, at a cost that does not grow with node count beyond building the
histograms.

Weighting
---------

Spatial means are weighted by **nodal volume**, not node count. Mesh density is
not uniform, so an unweighted mean would over-represent finely meshed regions.
Volume weighting is also what the extracellular potential integral will need in
the forward model, so the same weights carry into Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ionic_model as im
from .eikonal import ActivationResult, Velocities, solve_activation
from .mesh import VentricularMesh, region_from_rho
from .templates import (TemplateBank, Heterogeneity, build_template_bank,
                        IKR_REPOL_SEARCH_MS)

# A node counts as participating above this fraction of the current's own global
# peak magnitude, matching ecg12sim.timeline.ACTIVE_FRACTION_OF_PEAK.
ACTIVE_FRACTION_OF_PEAK = 0.05


@dataclass
class ReducedCurrent:
    """Exact spatial reductions of one current over the whole mesh, per sample."""

    peak: np.ndarray            # (nt,) signed extreme across nodes
    mean: np.ndarray            # (nt,) volume-weighted mean across nodes
    active_frac: np.ndarray     # (nt,) fraction of nodes participating
    peak_depth: np.ndarray      # (nt,) transmural depth holding the peak


@dataclass
class MeshBeat:
    """One beat assembled on the anatomical mesh, in reduced form."""

    t: np.ndarray                       # (nt,) ms
    mesh: VentricularMesh
    bank: TemplateBank
    activation: ActivationResult
    ina: ReducedCurrent
    ikr: ReducedCurrent
    vm_mean: np.ndarray                 # (nt,) mV, volume-weighted
    repol_time: np.ndarray              # (n_nodes,) ms
    qrs_window: tuple[float, float]
    t_window: tuple[float, float]
    region_mean_ina: dict = field(default_factory=dict)
    region_mean_ikr: dict = field(default_factory=dict)
    region_ina: list = field(default_factory=list)
    region_ikr: list = field(default_factory=list)

    @property
    def qt(self) -> float:
        """Activation onset to last repolarisation, ms."""
        return float(self.t_window[1] - self.qrs_window[0])

    @property
    def total_activation_time(self) -> float:
        return self.activation.total_activation_time

    def summary(self) -> str:
        L = [
            "Reaction-eikonal beat on anatomical mesh",
            "-" * 62,
            f"nodes / tetrahedra            : {self.mesh.n_nodes:,} / "
            f"{self.mesh.n_tets:,}",
            f"templates (transmural depths) : {self.bank.n_templates}",
            f"beat samples                  : {self.t.size:,} at "
            f"{self.bank.dt:g} ms",
            f"total activation time         : "
            f"{self.total_activation_time:.1f} ms",
            f"QRS / activation window       : {self.qrs_window[0]:.1f} - "
            f"{self.qrs_window[1]:.1f} ms",
            f"T wave / repolarisation window: {self.t_window[0]:.1f} - "
            f"{self.t_window[1]:.1f} ms",
            f"QT interval                   : {self.qt:.0f} ms",
            "",
            f"I_Na peak (inward)  : {self.ina.peak.min():8.1f} pA/pF",
            f"I_Kr peak, repolarisation : "
            f"{float(self.bank.ikr_repol_peak()[0].max()):8.3f} pA/pF",
            f"I_Kr peak, incl. early transient : "
            f"{self.ikr.peak.max():8.3f} pA/pF",
        ]
        if self.region_ina:
            L.append("")
            L.append("Per-node peaks averaged within each region, directly")
            L.append("comparable with the 1-D strand's table in tracking.py:")
            L.append(f"{'region':9s} {'I_Na peak':>11s} {'t_peak':>8s} | "
                     f"{'I_Kr peak':>10s} {'t_peak':>8s} {'I_Kr charge':>12s}")
            for a, k in zip(self.region_ina, self.region_ikr):
                L.append(f"{a.label:9s} {a.peak:11.1f} {a.t_peak:8.1f} | "
                         f"{k.peak:10.3f} {k.t_peak:8.1f} {k.charge:12.1f}")
        return "\n".join(L)


def nodal_volumes(mesh: VentricularMesh) -> np.ndarray:
    """Distribute tetrahedron volumes equally to their four vertices.

    Gives each node the share of myocardium it represents, so spatial averages
    are not biased by local mesh refinement.
    """
    vol = mesh.tet_volumes_cm3()
    w = np.zeros(mesh.n_nodes)
    np.add.at(w, mesh.tets.ravel(), np.repeat(vol / 4.0, 4))
    return w


def _shift_accumulate(template: np.ndarray, weights: np.ndarray,
                      nt: int) -> np.ndarray:
    """Sum of ``weights[s] * template`` shifted by each ``s``, exactly.

    This is the convolution ``template * weights`` truncated to ``nt`` samples,
    evaluated as a sparse accumulation because ``weights`` has non-zero entries
    only where nodes actually activate (a few hundred bins out of thousands).
    """
    out = np.zeros(nt)
    shifts = np.flatnonzero(weights)
    for s in shifts:
        seg = template[:max(nt - s, 0)]
        if seg.size:
            out[s:s + seg.size] += weights[s] * seg
    return out


def _shift_extreme(template: np.ndarray, present: np.ndarray, nt: int,
                   inward: bool) -> np.ndarray:
    """Extreme of ``template`` over all shifts present, per sample.

    A grayscale dilation (or erosion, for inward currents) of the template by
    the set of activation times. Unlike the weighted sum this ignores how many
    nodes sit at each activation time, because a maximum does not care.
    """
    fill = np.inf if inward else -np.inf
    out = np.full(nt, fill)
    reduce = np.minimum if inward else np.maximum
    for s in np.flatnonzero(present):
        seg = template[:max(nt - s, 0)]
        if seg.size:
            out[s:s + seg.size] = reduce(out[s:s + seg.size], seg)
    out[~np.isfinite(out)] = 0.0
    return out


def _reduce_current(tmpl: np.ndarray, bank: TemplateBank,
                    group: np.ndarray, at_idx: np.ndarray,
                    weights: np.ndarray, nt: int, inward: bool
                    ) -> ReducedCurrent:
    """Build exact spatial reductions of one current across the mesh."""
    n_groups = bank.n_templates
    total_w = float(weights.sum())
    n_nodes = int(group.size)

    threshold = ACTIVE_FRACTION_OF_PEAK * (abs(tmpl.min()) if inward
                                           else abs(tmpl.max()))

    mean = np.zeros(nt)
    active = np.zeros(nt)
    best = np.full(nt, np.inf if inward else -np.inf)
    best_group = np.full(nt, -1, dtype=int)

    for g in range(n_groups):
        sel = group == g
        if not sel.any():
            continue
        idx = at_idx[sel]

        # Volume-weighted and unweighted activation histograms for this depth.
        w_hist = np.bincount(idx, weights=weights[sel], minlength=nt)[:nt]
        n_hist = np.bincount(idx, minlength=nt)[:nt].astype(float)

        mean += _shift_accumulate(tmpl[g], w_hist, nt)

        indicator = ((tmpl[g] <= -threshold) if inward
                     else (tmpl[g] >= threshold)).astype(float)
        active += _shift_accumulate(indicator, n_hist, nt)

        ext = _shift_extreme(tmpl[g], n_hist, nt, inward)
        better = (ext < best) if inward else (ext > best)
        best_group = np.where(better, g, best_group)
        best = np.where(better, ext, best)

    best[~np.isfinite(best)] = 0.0
    depth = np.where(best_group >= 0, bank.rho[np.maximum(best_group, 0)], np.nan)
    # A sample where nothing is participating has no meaningful peak location.
    depth = np.where(active > 0, depth, np.nan)

    return ReducedCurrent(peak=best, mean=mean / max(total_w, 1e-30),
                          active_frac=active / max(n_nodes, 1),
                          peak_depth=depth)


def _region_node_stats(tmpl: np.ndarray, bank: TemplateBank,
                       group: np.ndarray, at: np.ndarray,
                       weights: np.ndarray, node_region: np.ndarray,
                       inward: bool, peak_from_ms: float = 0.0) -> list:
    """Per-node peaks averaged within each region.

    Deliberately mirrors :func:`ecg12sim.tracking._region_stats`: each node's
    *own* peak and peak time are taken first, then averaged over the region.
    Averaging the traces first would be wrong for I_Na, because neighbouring
    nodes fire at different times and a mean trace smears a ~1 ms spike into
    something an order of magnitude too small. (That smeared quantity is still
    available as ``region_mean_ina``, which is what the ECG integrates.)

    Averages are weighted by nodal volume, so a finely meshed region does not
    count for more than the myocardium it represents.
    """
    from .tracking import RegionCurrentStats

    # Each template's own peak, peak time and charge, computed once.
    # ``peak_from_ms`` skips I_Kr's early transient; see
    # ecg12sim.templates.IKR_REPOL_SEARCH_MS.
    idx = bank.peak_index(tmpl, inward=inward, t_min=peak_from_ms)
    peak_per_tmpl = tmpl[np.arange(tmpl.shape[0]), idx]
    t_peak_per_tmpl = bank.t[idx]
    charge_per_tmpl = np.trapezoid(tmpl, bank.t, axis=1)

    out = []
    for code, label in ((im.ENDO, "endo"), (im.MID, "mid (M)"), (im.EPI, "epi")):
        sel = (node_region == code) & (weights > 0)
        if not sel.any():
            continue
        g = group[sel]
        w = weights[sel]
        w_tot = float(w.sum())
        out.append(RegionCurrentStats(
            label=label,
            peak=float((peak_per_tmpl[g] * w).sum() / w_tot),
            # Absolute time of each node's peak is its activation time plus the
            # template's internal peak offset.
            t_peak=float(((at[sel] + t_peak_per_tmpl[g]) * w).sum() / w_tot),
            charge=float((charge_per_tmpl[g] * w).sum() / w_tot),
        ))
    return out


def _region_mean(tmpl: np.ndarray, bank: TemplateBank, group: np.ndarray,
                 at_idx: np.ndarray, weights: np.ndarray, nt: int,
                 node_region: np.ndarray) -> dict:
    """Volume-weighted mean of a current within each transmural region."""
    out = {}
    for code, label in ((im.ENDO, "endo"), (im.MID, "mid"), (im.EPI, "epi")):
        in_region = node_region == code
        if not in_region.any():
            continue
        acc = np.zeros(nt)
        w_tot = float(weights[in_region].sum())
        for g in range(bank.n_templates):
            sel = in_region & (group == g)
            if not sel.any():
                continue
            w_hist = np.bincount(at_idx[sel], weights=weights[sel],
                                 minlength=nt)[:nt]
            acc += _shift_accumulate(tmpl[g], w_hist, nt)
        out[label] = acc / max(w_tot, 1e-30)
    return out


def run_mesh_beat(mesh: VentricularMesh,
                  drug: im.Drug | None = None,
                  vel: Velocities | None = None,
                  bank: TemplateBank | None = None,
                  het: Heterogeneity | None = None,
                  n_templates: int = 15,
                  template_duration: float = 600.0,
                  activation: ActivationResult | None = None) -> MeshBeat:
    """Assemble one beat on the mesh and return its reduced form."""
    drug = drug or im.Drug()
    act = activation if activation is not None else solve_activation(mesh, vel)
    bank = bank or build_template_bank(drug=drug, n_templates=n_templates,
                                       duration=template_duration, het=het)

    at = act.activation_time
    reachable = np.isfinite(at)
    if not reachable.any():
        raise ValueError("no nodes activated; cannot assemble a beat")

    dt = bank.dt
    at0 = float(at[reachable].min())
    # Beat spans the activation spread plus one full template.
    nt = int(np.ceil((at[reachable].max() - at0) / dt)) + bank.t.size
    t = np.arange(nt) * dt + at0

    at_idx = np.zeros(at.size, dtype=np.int64)
    at_idx[reachable] = np.round((at[reachable] - at0) / dt).astype(np.int64)
    at_idx = np.clip(at_idx, 0, nt - 1)

    group = bank.index_for(mesh.rho.astype(float))
    weights = nodal_volumes(mesh)
    # Unreachable nodes contribute nothing.
    weights = np.where(reachable, weights, 0.0)

    node_region = region_from_rho(mesh.rho)

    ina = _reduce_current(bank.I_Na, bank, group, at_idx, weights, nt,
                          inward=True)
    ikr = _reduce_current(bank.I_Kr, bank, group, at_idx, weights, nt,
                          inward=False)

    vm_mean = np.zeros(nt)
    total_w = float(weights.sum())
    for g in range(bank.n_templates):
        sel = (group == g) & reachable
        if not sel.any():
            continue
        w_hist = np.bincount(at_idx[sel], weights=weights[sel],
                             minlength=nt)[:nt]
        vm_mean += _shift_accumulate(bank.Vm[g], w_hist, nt)
    # Nodes not yet activated still sit at resting potential, so the mean needs
    # the resting baseline restored for the share of the wall still at rest.
    rest = float(bank.Vm[:, 0].mean())
    activated_w = _shift_accumulate(
        np.ones(bank.t.size),
        np.bincount(at_idx[reachable], weights=weights[reachable],
                    minlength=nt)[:nt], nt)
    vm_mean = (vm_mean + rest * np.maximum(total_w - activated_w, 0.0)) \
        / max(total_w, 1e-30)

    repol = np.full(at.size, np.nan)
    repol[reachable] = at[reachable] + bank.repol_time[group[reachable]]

    qrs_window = (float(at[reachable].min()), float(at[reachable].max()))
    finite_repol = repol[np.isfinite(repol)]
    t_window = ((float(finite_repol.min()), float(finite_repol.max()))
                if finite_repol.size else (np.nan, np.nan))

    return MeshBeat(
        t=t, mesh=mesh, bank=bank, activation=act, ina=ina, ikr=ikr,
        vm_mean=vm_mean, repol_time=repol,
        qrs_window=qrs_window, t_window=t_window,
        region_mean_ina=_region_mean(bank.I_Na, bank, group, at_idx, weights,
                                     nt, node_region),
        region_mean_ikr=_region_mean(bank.I_Kr, bank, group, at_idx, weights,
                                     nt, node_region),
        region_ina=_region_node_stats(bank.I_Na, bank, group, at, weights,
                                      node_region, inward=True),
        region_ikr=_region_node_stats(bank.I_Kr, bank, group, at, weights,
                                      node_region, inward=False,
                                      peak_from_ms=IKR_REPOL_SEARCH_MS),
    )


def mesh_band_profile(beat: MeshBeat, n_bands: int = 9) -> list:
    """Transmural profile of both currents across the anatomical wall.

    Mirrors :func:`ecg12sim.tracking.cable_band_profile` so the two models'
    numbers are directly comparable. Each node's own peak comes from its
    depth-indexed template and its own peak *time* from that template offset by
    its activation time, which is where the spatial variation actually lives on
    the mesh. Averages are volume-weighted.
    """
    from .tracking import band_profile

    bank = beat.bank
    at = beat.activation.activation_time
    reach = np.isfinite(at)
    group = bank.index_for(beat.mesh.rho.astype(float))
    weights = np.where(reach, nodal_volumes(beat.mesh), 0.0)

    rows = np.arange(bank.n_templates)
    i_na = bank.peak_index(bank.I_Na, inward=True)
    i_kr = bank.peak_index(bank.I_Kr, inward=False,
                           t_min=IKR_REPOL_SEARCH_MS)

    ina_peak = bank.I_Na[rows, i_na][group]
    ikr_peak = bank.I_Kr[rows, i_kr][group]
    ina_t = np.where(reach, at + bank.t[i_na][group], np.nan)
    ikr_t = np.where(reach, at + bank.t[i_kr][group], np.nan)

    return band_profile(depth=beat.mesh.rho.astype(float),
                        ina_peak=ina_peak, ina_t=ina_t,
                        ikr_peak=ikr_peak, ikr_t=ikr_t,
                        weights=weights, n_bands=n_bands)
