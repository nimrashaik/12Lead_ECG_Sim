"""Anisotropic activation-time solve on a ventricular mesh.

This is the propagation backbone of the mesh model. Rather than integrating a
reaction-diffusion system over ~350k nodes for 25,000 time steps (hours in
NumPy, and tens of gigabytes to record), it solves directly for *when* each node
activates, which is the only thing the reaction-eikonal construction needs.

The anisotropic eikonal equation
--------------------------------

Activation time ``T`` satisfies

    sqrt( grad(T) . D . grad(T) ) = 1

with the velocity tensor built from the local orthotropic basis the cohort
supplies per tetrahedron:

    D = v_f^2 (f (x) f) + v_s^2 (s (x) s) + v_n^2 (n (x) n)

so the propagation speed along a unit direction ``d`` is

    v(d) = sqrt( v_f^2 (d.f)^2 + v_s^2 (d.s)^2 + v_n^2 (d.n)^2 )

Fibre-aligned propagation is fastest; transverse is roughly a third as fast in
human ventricle, and that anisotropy is what shapes the activation sequence (and
therefore the QRS) far more than wall geometry alone.

Method, and its known bias
--------------------------

Arrival times come from a single-source shortest-path solve over the mesh graph:
each tetrahedron edge is weighted by ``length / v(edge direction)``, and
``scipy.sparse.csgraph.dijkstra`` propagates from the stimulus set.

This is the graph-eikonal (Bellman fixed-point) formulation, chosen deliberately
over a hand-written Fast Iterative Method. The local tet solver an FIM needs is
a well-known source of subtle errors that are hard to detect from the output,
whereas Dijkstra's correctness reduces to the correctness of the edge weights,
and scipy's implementation is compiled and fast.

Its one systematic cost is **metrication bias**: paths are restricted to mesh
edges, so they cannot cut straight through a tetrahedron, and arrival times are
therefore biased *late* (effective velocity biased low) by a few percent on an
unstructured mesh. :func:`measure_effective_velocity` quantifies it instead of
leaving it implicit, and ``metrication_correction`` applies a calibration factor
when you want prescribed and realised velocity to agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .mesh import VentricularMesh

# The six vertex pairs forming the edges of a tetrahedron.
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass
class Velocities:
    """Conduction velocities along the local orthotropic axes, cm/s.

    Defaults are physiological for human ventricular myocardium. ``along_fibre``
    matches the 65 cm/s that the 1-D strand measures from its own I_Na
    wavefront, so the two models are parameterised consistently.
    """

    along_fibre: float = 65.0
    across_sheet: float = 25.0
    across_normal: float = 25.0

    #: Isotropic velocity inside a thin endocardial shell, cm/s. This stands in
    #: for the His-Purkinje system, which this model does not represent
    #: explicitly: without it, activation starting at one apical site has to
    #: crawl apex-to-base through working myocardium and takes roughly twice as
    #: long as a real heart (170 ms on cohort mesh 01, against a physiological
    #: 80-100 ms). A fast endocardial layer is the standard surrogate, and
    #: 150-200 cm/s is the usual range. The 150 cm/s default was chosen by
    #: sweeping against that target: on cohort mesh 01 it yields 80 ms total
    #: activation uncorrected, and 87 ms with the metrication correction (which
    #: divides the velocities, so it lengthens activation). Both sit inside the
    #: physiological window, so the model stays sane either way.
    #: Set to 0 to disable.
    #:
    #: Raised from 150 to 600 once the forward model existed to check it
    #: against. 150 cm/s leaves activation apex-to-base dominated, which
    #: inverts the frontal-plane leads (aVR positive, lead II negative). At
    #: 600 cm/s the endocardium activates nearly together, transmural
    #: propagation dominates, and the QRS axis comes out left-inferior as it
    #: should. The trade-off is total activation shortening to ~54 ms, below
    #: the 80-100 ms physiological window, so this value is calibrated for
    #: lead morphology rather than for activation duration. It is a surrogate
    #: for a whole Purkinje network, not a myocardial velocity.
    endo_layer_cv: float = 600.0

    #: Transmural depth defining that shell (0 = endocardium).
    endo_layer_rho: float = 0.10

    #: Divides the prescribed velocities to offset graph metrication bias.
    #: 1.0 leaves them untouched; see :func:`calibrate_metrication`.
    metrication_correction: float = 1.0

    def as_tuple(self) -> tuple[float, float, float]:
        c = self.metrication_correction
        return (self.along_fibre / c, self.across_sheet / c,
                self.across_normal / c)

    @property
    def endo_cv_corrected(self) -> float:
        return self.endo_layer_cv / self.metrication_correction


@dataclass
class ActivationResult:
    """Outcome of the activation-time solve."""

    activation_time: np.ndarray      # (n_nodes,) ms, inf where unreachable
    stimulus_nodes: np.ndarray       # (n_stim,) node indices
    velocities: Velocities
    n_unreachable: int

    @property
    def total_activation_time(self) -> float:
        """Time for the wavefront to cover the reachable myocardium, ms."""
        finite = self.activation_time[np.isfinite(self.activation_time)]
        return float(finite.max() - finite.min()) if finite.size else float("nan")

    def percentile_time(self, pct: float) -> float:
        finite = self.activation_time[np.isfinite(self.activation_time)]
        return float(np.percentile(finite, pct)) if finite.size else float("nan")


def _orthotropic_basis(mesh: VentricularMesh) -> tuple[np.ndarray, ...]:
    """Return per-tet (fibre, sheet, normal) unit vectors as float64."""
    f = mesh.fibres.astype(np.float64)
    s = mesh.sheets.astype(np.float64)

    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    # Re-orthogonalise the sheet axis against the fibre axis (Gram-Schmidt) so
    # the basis stays orthonormal after the float32 round-trip through the file.
    s -= f * np.einsum("ij,ij->i", s, f)[:, None]
    s_norm = np.linalg.norm(s, axis=1, keepdims=True)
    degenerate = (s_norm < 1e-8).ravel()
    if degenerate.any():
        # Fall back to any axis not parallel to the fibre direction.
        helper = np.tile(np.array([0.0, 0.0, 1.0]), (degenerate.sum(), 1))
        alt = np.cross(f[degenerate], helper)
        alt_norm = np.linalg.norm(alt, axis=1, keepdims=True)
        still_bad = (alt_norm < 1e-8).ravel()
        if still_bad.any():
            idx = np.flatnonzero(degenerate)[still_bad]
            alt[still_bad] = np.cross(f[idx], np.array([1.0, 0.0, 0.0]))
            alt_norm = np.linalg.norm(alt, axis=1, keepdims=True)
        s[degenerate] = alt / np.maximum(alt_norm, 1e-12)
        s_norm = np.linalg.norm(s, axis=1, keepdims=True)
    s /= np.maximum(s_norm, 1e-12)

    n = np.cross(f, s)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return f, s, n


def build_edge_graph(mesh: VentricularMesh, vel: Velocities
                     ) -> tuple[coo_matrix, np.ndarray]:
    """Build the symmetric travel-time graph over mesh edges.

    Each tetrahedron contributes six weighted edges. Where several tetrahedra
    share an edge they generally disagree (their fibre directions differ), so
    the **minimum** travel time wins: the wavefront takes the fastest available
    route, which is what the eikonal solution describes.

    Returns the graph in COO form plus the per-edge travel times in ms.
    """
    v_f, v_s, v_n = vel.as_tuple()
    pts = mesh.points.astype(np.float64)
    f, s, n = _orthotropic_basis(mesh)
    tets = mesh.tets

    rows = np.empty(tets.shape[0] * len(_TET_EDGES), dtype=np.int64)
    cols = np.empty_like(rows)
    times = np.empty(rows.size, dtype=np.float64)

    span = tets.shape[0]
    for k, (a, b) in enumerate(_TET_EDGES):
        i = tets[:, a].astype(np.int64)
        j = tets[:, b].astype(np.int64)
        d = pts[j] - pts[i]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1e-12)
        unit = d / safe[:, None]

        # v(d)^2 = sum over axes of v_axis^2 * (d . axis)^2
        speed_sq = (v_f ** 2 * np.einsum("ij,ij->i", unit, f) ** 2
                    + v_s ** 2 * np.einsum("ij,ij->i", unit, s) ** 2
                    + v_n ** 2 * np.einsum("ij,ij->i", unit, n) ** 2)
        speed = np.sqrt(np.maximum(speed_sq, 1e-12))      # cm/s

        # Fast endocardial shell: an edge lying wholly inside it conducts at the
        # shell velocity, isotropically, if that is faster than the fibre-based
        # value. Requiring *both* endpoints keeps the speed-up confined to the
        # shell instead of leaking one edge deep into the myocardium.
        if vel.endo_layer_cv > 0.0:
            endo_speed = vel.endo_cv_corrected
            in_shell = ((mesh.rho[i] <= vel.endo_layer_rho)
                        & (mesh.rho[j] <= vel.endo_layer_rho))
            speed = np.where(in_shell & (endo_speed > speed), endo_speed, speed)

        sl = slice(k * span, (k + 1) * span)
        rows[sl] = i
        cols[sl] = j
        # length[cm] / speed[cm/s] -> s, then to ms
        times[sl] = length / speed * 1000.0

    # Collapse duplicate edges to their fastest traversal.
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    key = lo * np.int64(mesh.n_nodes) + hi
    uniq, inverse = np.unique(key, return_inverse=True)
    best = np.full(uniq.size, np.inf)
    np.minimum.at(best, inverse, times)

    e_i = (uniq // np.int64(mesh.n_nodes)).astype(np.int32)
    e_j = (uniq % np.int64(mesh.n_nodes)).astype(np.int32)

    # Symmetric graph: both directions carry the same travel time.
    data = np.concatenate([best, best])
    r = np.concatenate([e_i, e_j])
    c = np.concatenate([e_j, e_i])
    graph = coo_matrix((data, (r, c)), shape=(mesh.n_nodes, mesh.n_nodes))
    return graph, best


@dataclass
class RootPoint:
    """A Purkinje-myocardial junction, placed in universal ventricular coords.

    ``ventricle`` is -1 for the left and +1 for the right, ``z`` runs 0 at the
    apex to 1 at the base, and ``phi`` is the rotational coordinate (0 at the
    septum, approaching +/- pi at the left-ventricular free wall). ``delay`` is
    the activation time of this junction in ms.
    """

    ventricle: int
    z: float
    phi: float
    delay: float = 0.0
    label: str = ""


#: Default earliest-activation sites, following the classic human mapping of
#: Durrer et al. (1970): activation breaks through first on the left-ventricular
#: paraseptal walls about a third of the way up from the apex, and on the
#: right-ventricular anterior wall near the septum, with the free walls
#: following slightly later.
#:
#: Distributed roots matter for lead morphology, not just speed. With a single
#: apical stimulus the wavefront crawls apex-to-base, so the net QRS dipole
#: points apex-to-base (superior and rightward) and the frontal-plane leads come
#: out inverted: aVR positive and lead II negative. Real Purkinje activation
#: reaches apex and base within roughly 15 ms of each other, which leaves the
#: transmural endocardium-to-epicardium component dominant and puts the QRS axis
#: down and to the left, where it belongs.
#: Measured QRS axis against root coverage and endocardial velocity
#: (body coordinates: left, anterior, superior; normal is left > 0, superior < 0):
#:
#:   roots     shell      total        axis
#:   sparse    150 cm/s   81.4 ms   (-0.25, -0.67, +0.70)
#:   sparse    400 cm/s   60.4 ms   (+0.06, -0.95, +0.30)
#:   full      150 cm/s   78.6 ms   (-0.11, -0.64, +0.76)
#:   full      300 cm/s   62.2 ms   (+0.11, -0.77, +0.63)
#:   full      400 cm/s   58.4 ms   (+0.22, -0.87, +0.44)
#:   full      600 cm/s   53.9 ms   (+0.43, -0.87, -0.25)   <- normal
#:
#: The axis moves monotonically with both, which is why the defaults sit at the
#: bottom row. The posterior component is correct, not an artefact: a normal QRS
#: axis does point posteriorly, which is what makes V1-V2 predominantly negative.
DURRER_ROOTS = (
    RootPoint(-1, 0.32, 0.65, 0.0, "LV anterior paraseptal"),
    RootPoint(-1, 0.28, -0.65, 0.0, "LV posterior paraseptal"),
    RootPoint(-1, 0.14, 0.00, 2.0, "LV apical septal"),
    RootPoint(-1, 0.45, 2.60, 10.0, "LV free wall"),
    RootPoint(+1, 0.38, 0.40, 5.0, "RV anterior paraseptal"),
    RootPoint(+1, 0.18, 0.00, 5.0, "RV apical"),
    RootPoint(+1, 0.45, 1.80, 12.0, "RV free wall"),
)

#: Fuller root set spanning the whole apico-basal range of both endocardial
#: surfaces, which is what the Purkinje network actually reaches. Basal coverage
#: is what removes the residual apex-to-base bias in the QRS axis, so this is
#: the default.
PURKINJE_ROOTS = (
    RootPoint(-1, 0.15, 0.00, 2.0, "LV apical septal"),
    RootPoint(-1, 0.32, 0.65, 0.0, "LV anterior paraseptal"),
    RootPoint(-1, 0.28, -0.65, 0.0, "LV posterior paraseptal"),
    RootPoint(-1, 0.50, 1.50, 6.0, "LV anterolateral"),
    RootPoint(-1, 0.50, -1.50, 6.0, "LV inferolateral"),
    RootPoint(-1, 0.70, 0.60, 10.0, "LV basal anterior"),
    RootPoint(-1, 0.70, -0.60, 10.0, "LV basal inferior"),
    RootPoint(-1, 0.55, 2.80, 12.0, "LV free wall"),
    RootPoint(+1, 0.18, 0.00, 5.0, "RV apical"),
    RootPoint(+1, 0.38, 0.40, 5.0, "RV anterior paraseptal"),
    RootPoint(+1, 0.55, 1.20, 10.0, "RV free wall"),
    RootPoint(+1, 0.70, 0.50, 14.0, "RV basal"),
)


def uvc_root_nodes(mesh: VentricularMesh, roots=PURKINJE_ROOTS,
                   rho_max: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Map root-point specifications onto the nearest endocardial mesh nodes.

    Returns ``(node_indices, delays)``. Matching happens in universal
    ventricular coordinates rather than Cartesian space, so the same
    specification lands in the anatomically equivalent spot on any cohort mesh.
    """
    idx = []
    delays = []
    for rp in roots:
        side = (mesh.v < 0) if rp.ventricle < 0 else (mesh.v > 0)
        cand = np.flatnonzero(side & (mesh.rho <= rho_max))
        if cand.size == 0:
            continue
        dz = mesh.z[cand].astype(float) - rp.z
        # phi is periodic on [-pi, pi]
        dphi = np.angle(np.exp(1j * (mesh.phi[cand].astype(float) - rp.phi)))
        # Weight the rotational mismatch down a little: z is the better-behaved
        # coordinate and phi is degenerate near the apex.
        cost = dz ** 2 + (0.5 * dphi / np.pi) ** 2
        idx.append(int(cand[np.argmin(cost)]))
        delays.append(float(rp.delay))

    if not idx:
        raise ValueError("no root points could be placed on this mesh")
    return np.array(idx, dtype=np.int64), np.array(delays, dtype=float)


def endocardial_stimulus(mesh: VentricularMesh, rho_max: float = 0.15,
                         z_max: float = 0.33) -> np.ndarray:
    """Nodes forming the initial activation site.

    Replicates the protocol the cohort's own authors used for their
    electrophysiology runs: the apical third of the endocardium
    (``Z < 0.33``), which stands in for the His-Purkinje system's earliest
    breakthrough without modelling the conduction system explicitly.
    """
    sel = (mesh.rho <= rho_max) & (mesh.z <= z_max)
    nodes = np.flatnonzero(sel)
    if nodes.size == 0:
        raise ValueError(
            f"no nodes with rho <= {rho_max} and z <= {z_max}; "
            f"rho range {mesh.rho.min():.2f}-{mesh.rho.max():.2f}, "
            f"z range {mesh.z.min():.2f}-{mesh.z.max():.2f}")
    return nodes


def solve_activation(mesh: VentricularMesh, vel: Velocities | None = None,
                     stimulus: np.ndarray | None = None,
                     delays: np.ndarray | None = None,
                     roots=PURKINJE_ROOTS) -> ActivationResult:
    """Solve for per-node activation time, in ms from earliest activation.

    With no explicit ``stimulus``, activation starts from the distributed root
    points in ``roots`` (see :data:`DURRER_ROOTS`). Pass ``roots=None`` to fall
    back to the apical endocardial patch, which is the cohort authors' own
    protocol but produces an apex-to-base dominated sequence.

    Per-source ``delays`` are handled by attaching one virtual source node to
    each root with an edge weight equal to that root's delay, then running a
    single-source shortest path from it. That is exact and costs one Dijkstra
    call rather than one per root.
    """
    vel = vel or Velocities()

    if stimulus is None:
        if roots:
            stim, stim_delays = uvc_root_nodes(mesh, roots)
        else:
            stim = endocardial_stimulus(mesh)
            stim_delays = np.zeros(stim.size)
    else:
        stim = np.asarray(stimulus, dtype=np.int64)
        stim_delays = (np.zeros(stim.size) if delays is None
                       else np.asarray(delays, dtype=float))

    graph, _ = build_edge_graph(mesh, vel)

    if np.any(stim_delays != 0.0):
        n = mesh.n_nodes
        src = n                                  # virtual super-source
        g = graph.tocoo()
        rows = np.concatenate([g.row, np.full(stim.size, src), stim])
        cols = np.concatenate([g.col, stim, np.full(stim.size, src)])
        data = np.concatenate([g.data, stim_delays, stim_delays])
        big = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
        at = dijkstra(big, directed=False, indices=src, min_only=True)[:n]
    else:
        at = dijkstra(graph.tocsr(), directed=False, indices=stim,
                      min_only=True)

    finite = np.isfinite(at)
    if finite.any():
        at = at - at[finite].min()               # measure from earliest onset

    return ActivationResult(
        activation_time=at,
        stimulus_nodes=stim,
        velocities=vel,
        n_unreachable=int(np.isinf(at).sum()),
    )


def measure_metrication(mesh: VentricularMesh, speed: float = 65.0,
                        source: int | None = None,
                        min_dist: float = 1.0, max_dist: float = 4.0) -> dict:
    """Quantify how far graph-restricted paths detour, using a known answer.

    Measuring this needs a configuration whose exact solution is known, so the
    probe is deliberately **isotropic**: with one uniform speed and a single
    point source, the true arrival time at any node is just straight-line
    distance over speed. Dijkstra can only do worse, because its paths hop from
    vertex to vertex instead of cutting through tetrahedra, so

        ratio = (arrival_time * speed) / straight_line_distance

    is exactly the path-length inflation, and is >= 1 by construction.

    Nodes are restricted to a mid-range distance band: very near the source the
    discretisation dominates, and very far the path may legitimately curve
    around concave geometry, which is not metrication error.

    Comparing random *anisotropic* node pairs cannot measure this. Two distant
    nodes often activate near-simultaneously via different routes, so
    distance/dt for arbitrary pairs is unbounded and means nothing.
    """
    iso = Velocities(speed, speed, speed, 1.0)
    if source is None:
        # A node near the mesh centroid, so the distance band stays inside tissue.
        centroid = mesh.points.astype(np.float64).mean(axis=0)
        source = int(np.argmin(
            np.linalg.norm(mesh.points.astype(np.float64) - centroid, axis=1)))

    res = solve_activation(mesh, iso, stimulus=np.array([source]))
    at = res.activation_time

    pts = mesh.points.astype(np.float64)
    dist = np.linalg.norm(pts - pts[source], axis=1)          # cm
    band = (dist >= min_dist) & (dist <= max_dist) & np.isfinite(at)
    if not band.any():
        return {}

    travelled = at[band] * speed / 1000.0                     # cm along the graph
    ratio = travelled / dist[band]
    return {
        "probe_speed": speed,
        "n_nodes": int(band.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "effective_speed_median": float(speed / np.median(ratio)),
    }


def calibrate_metrication(mesh: VentricularMesh, speed: float = 65.0) -> float:
    """Return the factor that makes realised velocity match the prescribed one.

    Feed the result back as ``Velocities.metrication_correction``. Because the
    probe is isotropic the factor is a property of the mesh, not of a particular
    anisotropy setting, so one calibration serves any velocity choice.
    """
    stats = measure_metrication(mesh, speed=speed)
    ratio = stats.get("ratio_median")
    return float(ratio) if ratio and np.isfinite(ratio) else 1.0


def format_activation_report(mesh: VentricularMesh, res: ActivationResult,
                             stats: dict | None = None) -> str:
    """Render the activation solve as a readable summary."""
    at = res.activation_time
    finite = at[np.isfinite(at)]
    reg = mesh.region()

    L = [
        "Anisotropic activation solve (graph eikonal)",
        "-" * 62,
        f"prescribed CV along fibre / transverse : "
        f"{res.velocities.along_fibre:.0f} / {res.velocities.across_sheet:.0f} cm/s",
        f"fast endocardial shell                 : "
        + (f"{res.velocities.endo_layer_cv:.0f} cm/s for rho <= "
           f"{res.velocities.endo_layer_rho:.2f}"
           if res.velocities.endo_layer_cv > 0 else "disabled"),
        f"metrication correction applied         : "
        f"{res.velocities.metrication_correction:.3f}",
        f"stimulus nodes (apical endocardium)    : {res.stimulus_nodes.size:,}",
        f"nodes activated                        : {finite.size:,} of {at.size:,}",
    ]
    if res.n_unreachable:
        L.append(f"UNREACHABLE nodes                      : {res.n_unreachable:,}")
    L += [
        f"total activation time                  : "
        f"{res.total_activation_time:.1f} ms",
        f"activation percentiles 10/50/90        : "
        f"{res.percentile_time(10):.1f} / {res.percentile_time(50):.1f} / "
        f"{res.percentile_time(90):.1f} ms",
    ]

    import ecg12sim.ionic_model as _im
    for code, label in ((_im.ENDO, "endo"), (_im.MID, "mid"), (_im.EPI, "epi")):
        sel = (reg == code) & np.isfinite(at)
        if sel.any():
            L.append(f"  mean activation, {label:4s}                 : "
                     f"{at[sel].mean():.1f} ms")

    if stats:
        L += [
            "",
            "Metrication check (isotropic probe, exact answer = distance/speed)",
            f"  path-length inflation, median        : {stats['ratio_median']:.4f}",
            f"  p05 - p95                            : "
            f"{stats['ratio_p05']:.4f} - {stats['ratio_p95']:.4f}",
            f"  effective vs probe speed             : "
            f"{stats['effective_speed_median']:.1f} vs "
            f"{stats['probe_speed']:.1f} cm/s",
            f"  nodes in distance band               : {stats['n_nodes']:,}",
        ]
    return "\n".join(L)
