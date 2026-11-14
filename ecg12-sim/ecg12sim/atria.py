"""Atrial myocardium from the same cohort mesh, and the P wave it generates.

:mod:`ecg12sim.mesh` keeps tags 1 and 2 of the Rodero four-chamber mesh (left-
and right-ventricular myocardium) and throws the rest away. This module takes up
tags **3 (left atrium) and 4 (right atrium)** from the very same file, so the
atria and the ventricles are the same subject's anatomy in one shared coordinate
system. That matters for the forward model: both chambers are seen by one set of
electrodes through one anatomical frame, with no registration step and nothing
to align.

What the cohort gives the atria, and what it does not
-----------------------------------------------------

The atria get **geometry and tissue tags only**. Two fields that the ventricular
model leans on are absent, and both absences are load-bearing for the design
here, so they are measured rather than assumed:

*Universal coordinates.* The published description states that the UVC system
"was assigned to the ventricles in the four-chamber meshes and all the other
labels were assigned with -10". Confirmed on case 01: every atrial node carries
RHO = PHI = Z = V = -10. So there is no transmural coordinate for the atria, and
none is invented here. The consequence is deliberate: the atria use a **single**
action-potential template rather than the ventricles' depth-indexed bank. The
wall justifies it, being only two to three elements thick (65% of atrial nodes
lie on a surface, against a median tetrahedron edge of 1.07 mm), so there is
little room for a transmural gradient even if a coordinate existed.

*Fibre architecture.* The cohort's own description says only that "ventricular
fibres were generated using a rule-based method", and the atrial fibre vectors
bear that out: measured over all 397,373 atrial tetrahedra, the orientation
tensor has eigenvalues (0, 0, 1) and the mean fibre vector has unit length,
meaning **every atrial fibre vector points in one identical direction**. It is a
placeholder, not an anatomical field. (The ventricular tags, for comparison,
give eigenvalues of roughly (0.26, 0.28, 0.46), i.e. a genuinely dispersed
field.) Conduction here is therefore **isotropic**: an anisotropic tensor built
from a constant direction would impose a fictitious uniform fibre axis across
both atria, which is worse than admitting the data is not there. The real cost
is that Bachmann's bundle and the other specialised fast pathways cannot be
represented, and :data:`ATRIAL_CV` records how that is absorbed.

Electrical isolation from the ventricles
---------------------------------------

The atrial and ventricular tag sets **share 5,480 nodes** in the mesh, around
the mitral and tricuspid annuli. Solving activation over the union would let the
atrial wavefront run straight into ventricular myocardium and depolarise it
directly, which is exactly what the annulus fibrosus prevents in a real heart:
the only electrical route from atrium to ventricle is the AV node and the His
bundle. So the two meshes are extracted and solved **separately**, and the only
coupling is a prescribed delay, the PR interval (see :mod:`ecg12sim.rhythm`).
That is both the anatomically correct topology and the simpler one.

Within the atria, by contrast, propagation is left to emerge: the left and right
atrial tag sets share 1,347 nodes across the interatrial septum, each chamber is
a single connected component, and so is their union. Interatrial conduction
therefore happens through tissue continuity rather than through any hand-placed
bridge.

Why the P wave carries fine structure
-------------------------------------

The simulated P wave is not perfectly smooth; it has low-amplitude undulation of
roughly 6-10% of its peak. That is anatomy rather than numerical noise, and the
distinction was measured rather than assumed. The signal is driven by how much
tissue depolarises per unit time, i.e. the occupancy of the activation-time
histogram, and that occupancy varies by about 57% (standard deviation over mean)
as the wavefront crosses the atrial shell: through narrow isthmuses, out into the
appendages, across the septum and around the vein ostia, the advancing wavefront
area genuinely changes.

The test that separates the two explanations is to coarsen the activation bins.
Rounding or counting noise would fall as the square root of the bin width;
genuine density variation would not move. Measured on case 01, relative jitter
goes 0.563, 0.575, 0.580, 0.579 for bins of 0.1, 0.5, 1.0 and 2.0 ms, i.e. flat.
At 0.5 ms there are already 513 nodes per bin, where counting noise would
contribute only 4.4%. So the variation is structural, and smoothing it away
would be cosmetic rather than a correction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import ionic_model as im
from . import templates as tmpl_mod
from .eikonal import ActivationResult, Velocities, build_edge_graph
from .mesh import MM_PER_CM, _pick, parse_vtk, tet_volumes_cm3

# Rodero cohort tissue tags.
TAG_LA_MYO = 3
TAG_RA_MYO = 4
ATRIAL_TAGS = (TAG_LA_MYO, TAG_RA_MYO)

#: Caval inlet tags, read only to locate the sinoatrial node.
TAG_SVC_INLET = 16
TAG_IVC_INLET = 17

CHAMBER_LA = -1
CHAMBER_RA = +1
CHAMBER_SEPTUM = 0

#: Isotropic atrial conduction velocity, cm/s.
#:
#: Measured conduction velocity in human atrial working myocardium is 60-100
#: cm/s. This default sits just above that range, and the gap is small enough to
#: be worth stating precisely rather than waving at. Measured on cohort case 01,
#: the longest activation path from the sinoatrial node is 11.0 cm, giving:
#:
#:     prescribed CV     total atrial activation
#:        80 cm/s              138 ms
#:       100 cm/s              110 ms
#:       116 cm/s               95 ms   <- default
#:       150 cm/s               73 ms
#:
#: Normal P-wave duration is 80-120 ms, so **100 cm/s, the top of the measured
#: physiological range, already lands inside the normal window at 110 ms**. The
#: model does not need a supraphysiological velocity to produce a normal P wave;
#: 116 cm/s is simply the value that centres total activation at
#: :data:`ATRIAL_ACTIVATION_TARGET_MS`, mid-window, instead of leaving it at the
#: upper edge.
#:
#: The residual 16% above measured tissue velocity is accounted for, not
#: absorbed silently: Bachmann's bundle conducts at roughly 120-180 cm/s and
#: carries activation from the right atrium to the left, and the crista
#: terminalis and pectinate muscles do the same within the right atrium. Every
#: one of those is a fibre-aligned structure, and as documented in this module's
#: header the cohort supplies no atrial fibre field, so none of them can be
#: represented. A single isotropic velocity has to cover both the working
#: myocardium and the fast pathways, so it lands between them. This is the same
#: kind of surrogate as :attr:`ecg12sim.eikonal.Velocities.endo_layer_cv` on the
#: ventricular side, but a far milder one: that parameter runs at 600 cm/s
#: against a myocardial 65.
#:
#: The default transfers across the cohort rather than being fitted to one mesh.
#: Calibrated on case 01, it was then applied unchanged to case 02 (84,728
#: atrial nodes against 97,985, and 43.3 cm^3 of myocardium against 49.9): total
#: atrial activation came out at 93.0 ms, against 94.9 ms on case 01, and case
#: 02's own calibration would have asked for 113.6 cm/s rather than 115.9. So
#: one velocity gives a normal P-wave duration on a mesh it was not tuned for.
#:
#: One further effect is real rather than numerical and is *not* corrected for.
#: The wavefront is confined to a 2.1 mm wall and must route around the blood
#: pool, so the graph geodesic reaches 1.23x the straight-line distance at long
#: range against 1.09x at short range, where metrication acts alone. The excess
#: is genuine anatomy: a straight line between distant atrial nodes passes
#: through the chamber cavity, which is blood, not tissue.
ATRIAL_CV = 116.0

#: Total atrial activation time the velocity is calibrated against, ms.
#: Normal P-wave duration is 80-120 ms; 95 ms sits mid-range.
ATRIAL_ACTIVATION_TARGET_MS = 95.0

_CACHE_VERSION = 1
_FIELDS = ("points", "tets", "tags", "fibres", "sheets", "chamber")
_LANDMARKS = ("svc_centroid", "ivc_centroid")


@dataclass
class AtrialMesh:
    """Biatrial myocardium reduced from a four-chamber cohort mesh.

    Coordinates are in **cm**, matching :class:`ecg12sim.mesh.VentricularMesh`
    and, critically, expressed in the *same* mesh frame, since both come from
    one file. The attribute names deliberately match the ventricular mesh where
    they mean the same thing, so :mod:`ecg12sim.eikonal` and
    :mod:`ecg12sim.forward` operate on either without special-casing.
    """

    points: np.ndarray        # (n_nodes, 3) cm
    tets: np.ndarray          # (n_tets, 4) int node indices
    tags: np.ndarray          # (n_tets,) 3 = LA myo, 4 = RA myo
    fibres: np.ndarray        # (n_tets, 3) placeholder, see module docstring
    sheets: np.ndarray        # (n_tets, 3) placeholder
    chamber: np.ndarray       # (n_nodes,) -1 LA, +1 RA, 0 shared septum
    svc_centroid: np.ndarray  # (3,) cm, superior vena cava inlet
    ivc_centroid: np.ndarray  # (3,) cm, inferior vena cava inlet
    source: str = ""

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    @property
    def rho(self) -> np.ndarray:
        """Placeholder transmural depth, constant across the atrial wall.

        The cohort assigns no universal coordinates to the atria (see the module
        docstring), and the atrial model uses one template regardless, so this
        exists only so the shared forward model can index a template bank. It is
        **not** a measured transmural depth, and nothing should read a
        transmural gradient out of it.
        """
        return np.full(self.n_nodes, 0.5, dtype=np.float32)

    @property
    def v(self) -> np.ndarray:
        """Alias of :attr:`chamber`, mirroring the ventricular mesh's ``v``."""
        return self.chamber

    @property
    def extent_cm(self) -> np.ndarray:
        return self.points.max(axis=0) - self.points.min(axis=0)

    def tet_volumes_cm3(self) -> np.ndarray:
        return tet_volumes_cm3(self.points, self.tets)

    def wall_thickness_mm(self) -> float:
        """Rough mean wall thickness, as volume over one side of the surface.

        A closed shell of area ``A`` and thickness ``h`` has volume ``A h`` and
        total surface ``2A``, so ``h = 2 V / S``. Reported to justify the
        single-template choice rather than to be used in any computation.
        """
        area = _surface_area_cm2(self.points, self.tets)
        if area <= 0:
            return float("nan")
        return 2.0 * float(self.tet_volumes_cm3().sum()) / area * MM_PER_CM

    def summary(self) -> str:
        vols = self.tet_volumes_cm3()
        ext = self.extent_cm
        n_la = int((self.tags == TAG_LA_MYO).sum())
        n_ra = int((self.tags == TAG_RA_MYO).sum())
        edge_mm = (float(np.median(vols)) * 6.0 * np.sqrt(2.0)) ** (1.0 / 3.0) * MM_PER_CM
        L = [
            "Atrial mesh",
            f"  source              : {self.source or 'unknown'}",
            f"  nodes / tetrahedra  : {self.n_nodes:,} / {self.n_tets:,}",
            f"  LA / RA tetrahedra  : {n_la:,} / {n_ra:,}",
            f"  septal (shared) nodes: "
            f"{int((self.chamber == CHAMBER_SEPTUM).sum()):,}",
            f"  bounding box (cm)   : {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}",
            f"  atrial myocardial volume : {vols.sum():.1f} cm^3",
            f"  median tet edge     : {edge_mm:.2f} mm",
            f"  mean wall thickness : {self.wall_thickness_mm():.2f} mm "
            f"(single template; no transmural coordinate exists)",
        ]
        return "\n".join(L)


def _surface_area_cm2(points: np.ndarray, tets: np.ndarray) -> float:
    """Area of the boundary surface: faces belonging to exactly one tet."""
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    faces = np.sort(faces, axis=1)
    uniq, counts = np.unique(faces, axis=0, return_counts=True)
    surf = uniq[counts == 1]
    if surf.size == 0:
        return 0.0
    p = points[surf].astype(np.float64)
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() / 2.0)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _tag_centroid_cm(raw: dict, tet_tags: np.ndarray, tag: int) -> np.ndarray:
    """Centroid of the nodes carrying one tissue tag, in cm."""
    sel = tet_tags == tag
    if not sel.any():
        return np.full(3, np.nan)
    nodes = np.unique(raw["tets"][sel])
    return (raw["points"][nodes].mean(axis=0) / MM_PER_CM).astype(np.float64)


def extract_atria(raw: dict, tags: tuple[int, ...] = ATRIAL_TAGS,
                  source: str = "") -> AtrialMesh:
    """Reduce a parsed four-chamber mesh to atrial myocardium.

    Mirrors :func:`ecg12sim.mesh.extract_ventricles`: keep the tagged
    tetrahedra, drop unreferenced nodes, renumber compactly. Additionally
    records the caval inlet centroids, because they locate the sinoatrial node
    and would otherwise be lost with the discarded tags.
    """
    tet_tags = _pick(raw, "cell", "ID", "id")
    if tet_tags is None:
        raise ValueError("mesh has no CELL_DATA 'ID' tag array")
    tet_tags = tet_tags.astype(int)

    keep = np.isin(tet_tags, tags)
    if not keep.any():
        raise ValueError(f"no tetrahedra carry atrial tags {tags}")

    # Landmarks must be read before the non-atrial tags are dropped.
    svc = _tag_centroid_cm(raw, tet_tags, TAG_SVC_INLET)
    ivc = _tag_centroid_cm(raw, tet_tags, TAG_IVC_INLET)

    kept_tets = raw["tets"][keep]
    used, remapped = np.unique(kept_tets, return_inverse=True)
    local = remapped.reshape(kept_tets.shape).astype(np.int32)
    kept_tags = tet_tags[keep]

    # Per-node chamber label. A node touched by both chambers' tetrahedra is on
    # the interatrial septum, which is the route interatrial conduction takes.
    in_la = np.zeros(used.size, dtype=bool)
    in_ra = np.zeros(used.size, dtype=bool)
    in_la[np.unique(local[kept_tags == TAG_LA_MYO])] = True
    in_ra[np.unique(local[kept_tags == TAG_RA_MYO])] = True
    chamber = np.where(in_la & in_ra, CHAMBER_SEPTUM,
                       np.where(in_la, CHAMBER_LA, CHAMBER_RA)).astype(np.int8)

    fibres = _pick(raw, "cell", "fibres", "fiber", "fibers")
    sheets = _pick(raw, "cell", "sheets", "sheet")
    n_kept = int(keep.sum())
    fibres = fibres[keep] if fibres is not None else np.zeros((n_kept, 3))
    sheets = sheets[keep] if sheets is not None else np.zeros((n_kept, 3))

    return AtrialMesh(
        points=(raw["points"][used] / MM_PER_CM).astype(np.float32),
        tets=local,
        tags=kept_tags.astype(np.int8),
        fibres=fibres.astype(np.float32),
        sheets=sheets.astype(np.float32),
        chamber=chamber,
        svc_centroid=svc,
        ivc_centroid=ivc,
        source=source,
    )


def fibre_field_is_placeholder(mesh: AtrialMesh, tol: float = 1e-3) -> bool:
    """Whether the fibre vectors are all one direction, i.e. carry no anatomy.

    Checks the orientation tensor's largest eigenvalue rather than the mean
    vector, so it is insensitive to fibre sign. A genuinely dispersed field
    gives roughly 0.46 on this cohort's ventricles; a single repeated direction
    gives exactly 1.
    """
    f = mesh.fibres.astype(np.float64)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    if not np.any(n > 0):
        return True
    f = f / np.maximum(n, 1e-12)
    T = np.einsum("ni,nj->ij", f, f) / f.shape[0]
    return bool(np.linalg.eigvalsh(T)[-1] > 1.0 - tol)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def save_cache(mesh: AtrialMesh, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path, _version=np.array([_CACHE_VERSION]),
        _source=np.array([mesh.source]),
        **{f: getattr(mesh, f) for f in _FIELDS},
        **{k: getattr(mesh, k) for k in _LANDMARKS})


def load_cache(path: str) -> AtrialMesh:
    with np.load(path, allow_pickle=False) as z:
        version = int(z["_version"][0]) if "_version" in z else 0
        if version != _CACHE_VERSION:
            raise ValueError(f"{path}: cache version {version}, "
                             f"expected {_CACHE_VERSION}")
        source = str(z["_source"][0]) if "_source" in z else ""
        return AtrialMesh(source=source,
                          **{f: z[f] for f in _FIELDS},
                          **{k: z[k] for k in _LANDMARKS})


def load_atrial_mesh(vtk_path: str, cache_path: str | None = None,
                     rebuild: bool = False) -> AtrialMesh:
    """Load atrial myocardium, using a cached reduction when available.

    Deliberately separate from
    :func:`ecg12sim.mesh.load_ventricular_mesh` rather than one combined loader:
    the two reductions are independent, each is cached on its own, and keeping
    them apart is what guarantees the atrial and ventricular activation solves
    cannot leak into one another through shared annulus nodes.
    """
    if cache_path is None:
        cache_path = os.path.splitext(vtk_path)[0] + ".atria.npz"

    if not rebuild and os.path.exists(cache_path):
        return load_cache(cache_path)

    raw = parse_vtk(vtk_path)
    mesh = extract_atria(raw, source=os.path.basename(vtk_path))
    save_cache(mesh, cache_path)
    return mesh


# ---------------------------------------------------------------------------
# Sinoatrial node and activation
# ---------------------------------------------------------------------------

def sinoatrial_node(mesh: AtrialMesh) -> tuple[int, float]:
    """Locate the sinoatrial node, returning ``(node_index, distance_cm)``.

    The SAN lies subepicardially at the junction of the superior vena cava with
    the right atrium, so it is placed as the right-atrial myocardial node
    nearest the SVC inlet centroid. Deriving it from a tag the cohort actually
    supplies keeps the site anatomical and lets it transfer between cases, where
    a hardcoded coordinate would not: the nearest right-atrial node lands
    0.49 cm from that centroid on case 01 and 0.73 cm on case 02, with no
    per-case adjustment.
    """
    if not np.all(np.isfinite(mesh.svc_centroid)):
        raise ValueError("mesh carries no SVC landmark; rebuild the atrial "
                         "cache from the .vtk so the inlet tag is read")
    ra = np.flatnonzero(mesh.chamber != CHAMBER_LA)
    if ra.size == 0:
        raise ValueError("mesh has no right-atrial nodes")
    d = np.linalg.norm(mesh.points[ra].astype(np.float64) - mesh.svc_centroid,
                       axis=1)
    k = int(np.argmin(d))
    return int(ra[k]), float(d[k])


def isotropic_velocities(cv: float = ATRIAL_CV) -> Velocities:
    """Velocity set for atrial conduction: isotropic, no fast endocardial shell.

    Isotropic because the cohort supplies no atrial fibre field, and with no
    shell because the endocardial Purkinje surrogate is a ventricular construct
    with no atrial counterpart. Expressed through the ventricular
    :class:`~ecg12sim.eikonal.Velocities` so that the already-validated
    :func:`~ecg12sim.eikonal.build_edge_graph` is reused unchanged: with the
    three axis velocities equal, the orthotropic tensor it builds reduces
    exactly to isotropy whatever the placeholder fibre direction happens to be.
    """
    return Velocities(along_fibre=cv, across_sheet=cv, across_normal=cv,
                      endo_layer_cv=0.0)


def solve_atrial_activation(mesh: AtrialMesh, cv: float = ATRIAL_CV,
                            san: int | None = None) -> ActivationResult:
    """Solve atrial activation times from a single sinoatrial source."""
    from scipy.sparse.csgraph import dijkstra

    vel = isotropic_velocities(cv)
    if san is None:
        san, _ = sinoatrial_node(mesh)

    graph, _ = build_edge_graph(mesh, vel)
    at = dijkstra(graph.tocsr(), directed=False,
                  indices=np.array([san], dtype=np.int64), min_only=True)

    finite = np.isfinite(at)
    if finite.any():
        at = at - at[finite].min()

    return ActivationResult(activation_time=at,
                            stimulus_nodes=np.array([san], dtype=np.int64),
                            velocities=vel,
                            n_unreachable=int(np.isinf(at).sum()))


def calibrate_atrial_cv(mesh: AtrialMesh,
                        target_ms: float = ATRIAL_ACTIVATION_TARGET_MS,
                        probe_cv: float = 100.0) -> float:
    """Velocity whose total atrial activation time equals ``target_ms``.

    Exact in one solve rather than iterative: activation time scales strictly
    inversely with an isotropic velocity, because every edge weight is
    ``length / cv``. So the ratio of the probe's activation time to the target
    scales the probe velocity directly.
    """
    res = solve_atrial_activation(mesh, cv=probe_cv)
    total = res.total_activation_time
    if not np.isfinite(total) or total <= 0:
        return probe_cv
    return float(probe_cv * total / target_ms)


def format_atrial_activation(mesh: AtrialMesh, res: ActivationResult) -> str:
    at = res.activation_time
    finite = at[np.isfinite(at)]
    san = int(res.stimulus_nodes[0])
    L = [
        "Atrial activation solve (isotropic graph eikonal)",
        "-" * 62,
        f"prescribed conduction velocity  : "
        f"{res.velocities.along_fibre:.0f} cm/s, isotropic "
        f"(measured atrial tissue: 60-100)",
        "  isotropic because the cohort supplies no atrial fibre field; the",
        "  margin above tissue velocity covers Bachmann's bundle and the other",
        "  fibre-aligned fast pathways, which therefore cannot be represented",
        f"sinoatrial node                 : node {san:,} at SVC junction, "
        f"{np.round(mesh.points[san], 2)} cm",
        f"nodes activated                 : {finite.size:,} of {at.size:,}",
    ]
    if res.n_unreachable:
        L.append(f"UNREACHABLE nodes               : {res.n_unreachable:,}")
    L += [
        f"total atrial activation time    : "
        f"{res.total_activation_time:.1f} ms  (normal P wave 80-120 ms)",
        f"activation percentiles 10/50/90 : "
        f"{res.percentile_time(10):.1f} / {res.percentile_time(50):.1f} / "
        f"{res.percentile_time(90):.1f} ms",
    ]
    for code, label in ((CHAMBER_RA, "right atrium"), (CHAMBER_LA, "left atrium"),
                        (CHAMBER_SEPTUM, "septum (shared)")):
        sel = (mesh.chamber == code) & np.isfinite(at)
        if sel.any():
            L.append(f"  {label:16s} activated {at[sel].min():6.1f} - "
                     f"{at[sel].max():6.1f} ms (mean {at[sel].mean():.1f})")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Atrial template and beat
# ---------------------------------------------------------------------------

#: Flat heterogeneity: the atrial template bank applies no regional conductance
#: scaling, because there is no atrial coordinate to vary anything along. These
#: parameter values make :class:`ecg12sim.templates.Heterogeneity` return
#: exactly 1.0 for both I_Ks and I_to at every depth.
FLAT_HETEROGENEITY = tmpl_mod.Heterogeneity(
    m_gks_dip=1.0, epi_gks_boost=0.0, endo_gto=1.0)

#: Quiescence integrated before the atrial template's stimulus, ms. Longer than
#: the ventricular 50 ms because the I_Kur inactivation gate ``ui`` starts at
#: 1.0 and has to relax to its resting value under the atrial conductance set.
ATRIAL_PREPACE_MS = 250.0


def build_atrial_template_bank(drug: im.Drug | None = None,
                               duration: float = 600.0,
                               dt: float = 0.02,
                               out_dt: float = 0.1) -> tmpl_mod.TemplateBank:
    """One atrial action-potential template, with its I_Na and I_Kr traces.

    A single template, because the atria have no transmural coordinate to index
    a bank by and a two-to-three-element wall to vary one across. Reuses the
    ventricular template machinery wholesale, so the atrial cell is integrated
    by the same ionic model, with the same I_Na and I_Kr formulations, and the
    project's current-tracking story extends to the atria unchanged. What makes
    it atrial is :data:`ecg12sim.ionic_model.ATRIAL_CONDUCTANCES`.
    """
    return tmpl_mod.build_template_bank(
        drug=drug or im.Drug(),
        cond=im.ATRIAL_CONDUCTANCES,
        n_templates=1,
        duration=duration,
        dt=dt,
        out_dt=out_dt,
        het=FLAT_HETEROGENEITY,
        prepace=ATRIAL_PREPACE_MS)


@dataclass
class AtrialBeat:
    """One atrial beat, shaped so the shared forward model accepts it.

    Carries exactly the attributes :mod:`ecg12sim.forward` reads from a
    :class:`ecg12sim.mesh_sim.MeshBeat` (``t``, ``mesh``, ``bank``,
    ``activation``), which is why the P wave needs no separate forward solver.
    """

    t: np.ndarray
    mesh: AtrialMesh
    bank: tmpl_mod.TemplateBank
    activation: ActivationResult
    p_window: tuple[float, float]       # atrial depolarisation span, ms
    ta_window: tuple[float, float]      # atrial repolarisation span, ms

    @property
    def total_activation_time(self) -> float:
        return self.activation.total_activation_time

    def summary(self) -> str:
        b = self.bank
        return "\n".join([
            "Atrial beat (reaction-eikonal, single template)",
            "-" * 62,
            f"nodes / tetrahedra      : {self.mesh.n_nodes:,} / "
            f"{self.mesh.n_tets:,}",
            f"beat samples            : {self.t.size:,} at {b.dt:g} ms",
            f"atrial activation span  : {self.p_window[0]:.1f} - "
            f"{self.p_window[1]:.1f} ms  "
            f"(= P-wave duration {self.p_window[1] - self.p_window[0]:.0f} ms)",
            f"atrial repolarisation   : {self.ta_window[0]:.1f} - "
            f"{self.ta_window[1]:.1f} ms",
            f"atrial APD90            : {float(b.apd90[0]):.1f} ms "
            f"(human atrium 250-300 ms)",
            f"atrial AP peak / rest   : {float(b.Vm[0].max()):.1f} / "
            f"{float(b.Vm[0, 0]):.1f} mV",
            f"atrial I_Na peak        : {float(b.I_Na.min()):.1f} pA/pF",
            f"atrial I_Kr repol peak  : "
            f"{float(b.ikr_repol_peak()[0].max()):.3f} pA/pF",
        ])


def run_atrial_beat(mesh: AtrialMesh,
                    drug: im.Drug | None = None,
                    cv: float = ATRIAL_CV,
                    bank: tmpl_mod.TemplateBank | None = None,
                    activation: ActivationResult | None = None,
                    template_duration: float = 600.0) -> AtrialBeat:
    """Assemble one atrial beat: activation sequence plus the atrial template."""
    act = activation if activation is not None else solve_atrial_activation(
        mesh, cv=cv)
    bank = bank or build_atrial_template_bank(drug=drug,
                                              duration=template_duration)

    at = act.activation_time
    reachable = np.isfinite(at)
    if not reachable.any():
        raise ValueError("no atrial nodes activated; cannot assemble a beat")

    at0 = float(at[reachable].min())
    nt = int(np.ceil((at[reachable].max() - at0) / bank.dt)) + bank.t.size
    t = np.arange(nt) * bank.dt + at0

    repol = at[reachable] + float(bank.repol_time[0])
    return AtrialBeat(
        t=t, mesh=mesh, bank=bank, activation=act,
        p_window=(float(at[reachable].min()), float(at[reachable].max())),
        ta_window=(float(np.nanmin(repol)), float(np.nanmax(repol))))
