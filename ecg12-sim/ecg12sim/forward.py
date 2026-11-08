"""Forward model: 3-D cardiac source to genuine 12-lead electrode potentials.

This replaces the vectorcardiographic *presentation layer* in
:mod:`ecg12sim.leads`. There, a single 1-D pseudo-ECG was split into two phases
and projected onto two hardcoded axis vectors through the inverse Dower matrix,
because a 1.485 cm strand has only one electrical axis and cannot produce twelve
independent signals. Here the source is a 3-D biventricular mesh, so each
electrode genuinely sees a different view of the heart and the standard limb and
precordial leads follow from real potential differences.

The physics
-----------

For a current source in an unbounded homogeneous medium of conductivity
``sigma_e``, the extracellular potential at electrode position ``x_e`` is the
dipole integral (the same Gima & Rudy relation that ``cable.py`` evaluates along
its strand, here in three dimensions):

    phi(x_e) = -(1 / 4 pi sigma_e) * integral( sigma_i grad(Vm) . grad(1/r) dV )

with ``r = |x_source - x_e|`` and ``sigma_i`` the anisotropic intracellular
conductivity tensor built from the local fibre direction.

On linear tetrahedra ``grad(Vm)`` is constant per element and expands over the
four barycentric shape functions, ``grad(Vm) = sum_i Vm_i grad(lambda_i)``.
Substituting makes the potential **linear in the nodal voltages**:

    phi_e(t) = sum_over_nodes  w[e, node] * Vm(node, t)

where the weights, the *lead field*, depend only on geometry and conductivity:

    w[e, node] = -(1 / 4 pi sigma_e)
                 * sum over tets containing node of
                   (sigma_i grad(lambda_node)) . (d / r^3) * volume

Why this composes with the reaction-eikonal model
-------------------------------------------------

Because ``phi_e`` is a weighted sum of nodal voltages, and every nodal voltage is
a template shifted by that node's activation time, the electrode signal is again
a convolution of templates with weighted activation-time histograms, exactly as
in :mod:`ecg12sim.mesh_sim`. So the twelve leads cost one pass over the
lead field plus a handful of short convolutions, and the result is exact rather
than sampled.

Honest scope
------------

The medium is **unbounded and homogeneous**. There is no torso boundary and no
separate lung, blood or skeletal-muscle conductivity, so absolute amplitudes are
underestimated and the real torso's shaping of the field is absent. Electrode
positions come from standard surface anatomy rather than a segmented torso. What
this does deliver, and the Dower projection did not, is twelve signals that are
genuinely independent functions of a 3-D anatomical source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import VentricularMesh

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

ELECTRODE_NAMES = ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---------------------------------------------------------------------------
# Anatomical orientation
# ---------------------------------------------------------------------------

# The heart's two principal axes expressed in body coordinates
# (x = toward the subject's left, y = anterior, z = superior).
#
# Standard cardiac anatomy: the apex points left-anterior-inferior and the base
# lies right-posterior-superior, giving a long axis roughly 45 degrees oblique;
# the right ventricle sits anterior and to the right of the left ventricle.
# These two reference directions are what the measured mesh axes get aligned to,
# so the orientation is derived from anatomy rather than tuned to make the leads
# come out right.
APEX_TO_BASE_BODY = np.array([-0.45, -0.55, 0.70])   # right, posterior, superior
LV_TO_RV_BODY = np.array([-0.75, 0.65, 0.00])        # right, anterior

#: Electrode sites in body coordinates, centimetres, with the origin in the
#: mid-sagittal plane at the level of the heart. Precordial sites follow the
#: clinical definitions (V1/V2 at the 4th intercostal space either side of the
#: sternum, V4 at the 5th space in the midclavicular line, V6 in the mid-axillary
#: line); limb electrodes sit at the shoulders and the left hip.
ELECTRODES_BODY = {
    "V1": (-2.0, 9.0, 2.0),
    "V2": (2.0, 9.0, 2.0),
    "V3": (5.5, 8.5, 0.0),
    "V4": (8.0, 7.5, -1.0),
    "V5": (11.0, 5.0, -1.0),
    "V6": (13.5, 1.0, -1.0),
    "RA": (-20.0, 2.0, 22.0),
    "LA": (20.0, 2.0, 22.0),
    "LL": (8.0, 2.0, -40.0),
}

#: Where the heart's centroid sits relative to that origin, centimetres:
#: left of the midline and slightly anterior.
HEART_OFFSET_BODY = np.array([2.5, 1.0, 0.0])


def _orthonormal_triad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Right-handed triad with column 0 along ``a`` and ``b`` in the 0-1 plane."""
    u1 = a / np.linalg.norm(a)
    rest = b - u1 * np.dot(b, u1)
    n = np.linalg.norm(rest)
    if n < 1e-12:
        raise ValueError("reference directions are parallel")
    u2 = rest / n
    u3 = np.cross(u1, u2)
    return np.column_stack([u1, u2, u3])


@dataclass
class AnatomicalFrame:
    """Mapping between mesh coordinates and body coordinates.

    Built by aligning two directions measured on the mesh (apex to base, and
    left ventricle to right ventricle) with the same two directions in standard
    body anatomy. Two independent vector pairs fix a rotation completely.
    """

    origin: np.ndarray            # (3,) heart centroid in mesh coords, cm
    rotation: np.ndarray          # (3,3) mesh -> body
    apex: np.ndarray
    base: np.ndarray
    lv: np.ndarray
    rv: np.ndarray
    axis_angle_deg: float         # measured angle between the two mesh axes

    def to_mesh(self, body_xyz) -> np.ndarray:
        """Body coordinates (cm, origin at mid-sagittal heart level) to mesh."""
        rel = np.asarray(body_xyz, dtype=float) - HEART_OFFSET_BODY
        return self.origin + self.rotation.T @ rel

    def body_axes_in_mesh(self) -> dict:
        """The body axes expressed as mesh-frame unit vectors."""
        return {name: self.rotation.T @ np.eye(3)[i]
                for i, name in enumerate(("left", "anterior", "superior"))}

    def summary(self) -> str:
        ax = self.body_axes_in_mesh()
        L = [
            "Anatomical frame (mesh -> body)",
            f"  heart centroid (mesh, cm) : {np.round(self.origin, 2)}",
            f"  apex -> base length       : "
            f"{np.linalg.norm(self.base - self.apex):.2f} cm",
            f"  LV -> RV length           : "
            f"{np.linalg.norm(self.rv - self.lv):.2f} cm",
            f"  angle between those axes  : {self.axis_angle_deg:.1f} deg "
            f"(anatomy implies ~91)",
        ]
        for name, v in ax.items():
            L.append(f"  body '{name}' in mesh dirs  : {np.round(v, 3)}")
        return "\n".join(L)


def build_frame(mesh: VentricularMesh, z_apex: float = 0.05,
                z_base: float = 0.95) -> AnatomicalFrame:
    """Derive the body frame from the mesh's universal ventricular coordinates."""
    p = mesh.points.astype(np.float64)

    sel_apex = mesh.z < z_apex
    sel_base = mesh.z > z_base
    sel_lv = mesh.v < -0.5
    sel_rv = mesh.v > 0.5
    for sel, what in ((sel_apex, "apex"), (sel_base, "base"),
                      (sel_lv, "LV"), (sel_rv, "RV")):
        if not sel.any():
            raise ValueError(f"cannot locate {what} from the UVC fields")

    apex = p[sel_apex].mean(axis=0)
    base = p[sel_base].mean(axis=0)
    lv = p[sel_lv].mean(axis=0)
    rv = p[sel_rv].mean(axis=0)

    a_mesh = base - apex
    b_mesh = rv - lv
    angle = float(np.degrees(np.arccos(
        np.clip(np.dot(a_mesh, b_mesh)
                / np.linalg.norm(a_mesh) / np.linalg.norm(b_mesh), -1, 1))))

    U = _orthonormal_triad(a_mesh, b_mesh)                  # mesh frame
    V = _orthonormal_triad(APEX_TO_BASE_BODY, LV_TO_RV_BODY)  # body frame
    rotation = V @ U.T

    return AnatomicalFrame(origin=p.mean(axis=0), rotation=rotation,
                           apex=apex, base=base, lv=lv, rv=rv,
                           axis_angle_deg=angle)


def electrode_positions(frame: AnatomicalFrame) -> dict:
    """Standard electrode sites mapped into mesh coordinates, cm."""
    return {name: frame.to_mesh(xyz) for name, xyz in ELECTRODES_BODY.items()}


# ---------------------------------------------------------------------------
# Lead field
# ---------------------------------------------------------------------------

@dataclass
class Conductivity:
    """Bidomain-style conductivities, S/m.

    Defaults are the conventional human ventricular values: intracellular
    conduction is roughly an order of magnitude easier along the fibre than
    across it, and the surrounding medium is treated as uniform.
    """

    sigma_i_long: float = 0.17
    sigma_i_trans: float = 0.019
    sigma_e: float = 0.20


@dataclass
class LeadField:
    """Per-electrode nodal weights turning nodal voltages into potentials."""

    weights: np.ndarray                 # (n_electrodes, n_nodes), mV -> mV
    names: list
    positions: dict
    conductivity: Conductivity
    frame: AnatomicalFrame = field(default=None)

    def summary(self) -> str:
        w = self.weights
        L = ["Lead field",
             f"  electrodes : {len(self.names)}",
             f"  nodes      : {w.shape[1]:,}",
             f"  |w| per electrode (sum of abs):"]
        for i, n in enumerate(self.names):
            L.append(f"    {n:3s} {np.abs(w[i]).sum():.4e}")
        return "\n".join(L)


def _shape_function_gradients(pts: np.ndarray, tets: np.ndarray):
    """Gradients of the four barycentric shape functions, and tet volumes.

    Returns ``(grads, volumes)`` with ``grads`` shaped ``(n, 4, 3)`` in 1/m and
    ``volumes`` in m^3. Positions arrive in cm and are converted here, so the
    lead field comes out in SI and needs no further scaling.
    """
    p = pts[tets] / 100.0                       # cm -> m, (n, 4, 3)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]

    J = np.stack([e1, e2, e3], axis=1)          # (n, 3, 3), rows are edges
    det = np.linalg.det(J)
    vol = np.abs(det) / 6.0

    # grad(lambda_1..3) are the rows of J^{-1} transposed; lambda_0 closes the sum.
    Jinv = np.linalg.inv(J)                     # (n, 3, 3)
    g123 = np.transpose(Jinv, (0, 2, 1))        # (n, 3, 3): [i] -> grad lambda_{i+1}
    g0 = -g123.sum(axis=1, keepdims=True)
    grads = np.concatenate([g0, g123], axis=1)  # (n, 4, 3)
    return grads, vol


def lead_field_weights(mesh: VentricularMesh, points_cm: np.ndarray,
                       cond: Conductivity | None = None,
                       chunk: int = 250_000) -> np.ndarray:
    """Unbounded-medium nodal weights for arbitrary observation points.

    Returns ``(n_points, n_nodes)``. Callers that need many observation points
    (the torso surface, for the boundary-element solve) must chunk over points
    themselves: the full matrix for a torso surface would be gigabytes.

    Processed in chunks of tetrahedra, since the shape-function gradients alone
    are ``n_tets x 4 x 3`` doubles.
    """
    cond = cond or Conductivity()
    elec = np.asarray(points_cm, dtype=float) / 100.0      # cm -> m

    n_nodes = mesh.n_nodes
    weights = np.zeros((elec.shape[0], n_nodes))
    pts = mesh.points.astype(np.float64)
    pre = -1.0 / (4.0 * np.pi * cond.sigma_e)

    for lo in range(0, mesh.n_tets, chunk):
        hi = min(lo + chunk, mesh.n_tets)
        tets = mesh.tets[lo:hi]
        grads, vol = _shape_function_gradients(pts, tets)

        f = mesh.fibres[lo:hi].astype(np.float64)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

        # sigma_i grad(lambda) for a transversely isotropic tensor
        #   sigma_i = s_t I + (s_l - s_t) f (x) f
        dot_f = np.einsum("nij,nj->ni", grads, f)                 # (n, 4)
        sg = (cond.sigma_i_trans * grads
              + (cond.sigma_i_long - cond.sigma_i_trans)
              * dot_f[:, :, None] * f[:, None, :])                # (n, 4, 3)

        centroid = (pts[tets] / 100.0).mean(axis=1)               # (n, 3) m
        flat = tets.ravel()

        for e, xe in enumerate(elec):
            d = xe[None, :] - centroid                            # (n, 3)
            r = np.linalg.norm(d, axis=1)
            kernel = d / np.maximum(r, 1e-9)[:, None] ** 3        # (n, 3)
            contrib = pre * np.einsum("nij,nj->ni", sg, kernel) * vol[:, None]
            weights[e] += np.bincount(flat, weights=contrib.ravel(),
                                      minlength=n_nodes)

    return weights


def build_lead_field(mesh: VentricularMesh, positions: dict,
                     cond: Conductivity | None = None,
                     chunk: int = 250_000,
                     frame: AnatomicalFrame = None) -> LeadField:
    """Unbounded-medium lead field for the nine standard electrode sites."""
    cond = cond or Conductivity()
    names = [n for n in ELECTRODE_NAMES if n in positions]
    pts = np.array([positions[n] for n in names], dtype=float)
    weights = lead_field_weights(mesh, pts, cond=cond, chunk=chunk)
    return LeadField(weights=weights, names=names, positions=positions,
                     conductivity=cond, frame=frame)


# ---------------------------------------------------------------------------
# Electrode potentials and the twelve leads
# ---------------------------------------------------------------------------

def electrode_potentials(beat, lf: LeadField) -> dict:
    """Electrode potentials over the beat, in mV.

    Uses the same convolution identity as :mod:`ecg12sim.mesh_sim`: each
    electrode's signal is a lead-field-weighted sum of nodal voltages, and every
    nodal voltage is a template shifted by that node's activation time, so the
    sum collapses to one short convolution per (electrode, template) pair.

    Nodes that have not yet activated still sit at resting potential, and that
    resting contribution is added back explicitly. Dropping it would leave a
    spurious drifting baseline, because the lead-field weights do not sum to
    zero for an electrode.
    """
    from .mesh_sim import nodal_volumes  # noqa: F401  (documents the weighting)

    bank = beat.bank
    mesh = beat.mesh
    at = beat.activation.activation_time
    reachable = np.isfinite(at)

    nt = beat.t.size
    dt = bank.dt
    at0 = float(at[reachable].min())
    at_idx = np.zeros(at.size, dtype=np.int64)
    at_idx[reachable] = np.round((at[reachable] - at0) / dt).astype(np.int64)
    at_idx = np.clip(at_idx, 0, nt - 1)

    group = bank.index_for(mesh.rho.astype(float))
    rest = float(bank.Vm[:, 0].mean())
    tmpl_len = bank.t.size

    out = {}
    for e, name in enumerate(lf.names):
        w = lf.weights[e]
        signal = np.zeros(nt)

        for g in range(bank.n_templates):
            sel = (group == g) & reachable
            if not sel.any():
                continue
            hist = np.bincount(at_idx[sel], weights=w[sel], minlength=nt)[:nt]
            tmpl = bank.Vm[g] - rest        # deviation from rest
            for s in np.flatnonzero(hist):
                seg = tmpl[:max(nt - s, 0)]
                if seg.size:
                    signal[s:s + seg.size] += hist[s] * seg

        # Constant resting contribution from every reachable node.
        signal += rest * w[reachable].sum()
        out[name] = signal
    return out


def derive_12_leads(phi: dict) -> dict:
    """Standard 12 leads from the nine electrode potentials.

    Limb leads are Einthoven differences, augmented leads are Goldberger
    combinations, and the precordial leads are referenced to the Wilson central
    terminal. No fitted axes, no transform matrix, no display boost: every lead
    is an actual potential difference between electrode sites.
    """
    ra, la, ll = phi["RA"], phi["LA"], phi["LL"]

    leads = {
        "I": la - ra,
        "II": ll - ra,
        "III": ll - la,
        "aVR": ra - 0.5 * (la + ll),
        "aVL": la - 0.5 * (ra + ll),
        "aVF": ll - 0.5 * (ra + la),
    }
    wct = (ra + la + ll) / 3.0
    for i in range(1, 7):
        name = f"V{i}"
        if name in phi:
            leads[name] = phi[name] - wct
    return {n: leads[n] for n in LEAD_NAMES if n in leads}


def run_12lead_mesh(beat, cond: Conductivity | None = None,
                    frame: AnatomicalFrame | None = None) -> tuple:
    """Convenience: frame, lead field, electrode potentials and twelve leads."""
    frame = frame or build_frame(beat.mesh)
    positions = electrode_positions(frame)
    lf = build_lead_field(beat.mesh, positions, cond=cond, frame=frame)
    phi = electrode_potentials(beat, lf)
    return frame, lf, phi, derive_12_leads(phi)


# ---------------------------------------------------------------------------
# Morphology checks
# ---------------------------------------------------------------------------

def check_morphology(leads: dict, qrs_window: tuple,
                     t_window: tuple, t: np.ndarray) -> list:
    """Test the leads against textbook features of a normal ECG.

    These are the real validation of the forward model's geometry. Lead polarity
    and precordial R-wave progression follow from where the electrodes sit
    relative to the heart's electrical axis, so if the anatomical frame were
    wrong these would fail. Each entry is (description, passed, detail).
    """
    out = []
    qrs = (t >= qrs_window[0]) & (t <= qrs_window[1])
    tw = (t >= t_window[0]) & (t <= t_window[1])

    def _net(name, mask):
        """Net signed area of the deflection, mV*ms.

        Area rather than the single largest excursion. QRS polarity is
        conventionally judged by the net integral, which is what relates to the
        mean electrical axis; picking the extreme sample flips sign arbitrarily
        on a biphasic complex (an RS pattern with a slightly deeper S reads as
        wholly negative), which made the earlier version of these checks
        unstable rather than informative.
        """
        sig = leads[name][mask]
        if sig.size < 2:
            return 0.0
        return float(np.trapezoid(sig, t[mask]))

    r_aVR = _net("aVR", qrs)
    out.append(("aVR QRS area is negative", r_aVR < 0, f"{r_aVR:+.2f} mV.ms"))

    r_II = _net("II", qrs)
    out.append(("II QRS area is positive", r_II > 0, f"{r_II:+.2f} mV.ms"))

    # R-wave progression: the precordial deflection should swing from
    # predominantly negative at V1 to predominantly positive at V6.
    nets = [_net(f"V{i}", qrs) for i in range(1, 7)]
    out.append(("V1 QRS more negative than V6",
                nets[0] < nets[-1],
                "V1 %+.3f -> V6 %+.3f mV" % (nets[0], nets[-1])))

    rising = sum(1 for a, b in zip(nets, nets[1:]) if b > a)
    out.append(("precordial progression mostly monotonic", rising >= 3,
                f"{rising} of 5 steps increasing: "
                + ", ".join("%+.2f" % v for v in nets)))

    t_II = _net("II", tw)
    out.append(("T wave upright in II", t_II > 0, f"{t_II:+.3f} mV"))

    t_V5 = _net("V5", tw)
    out.append(("T wave upright in V5", t_V5 > 0, f"{t_V5:+.3f} mV"))

    return out


def format_morphology(checks: list) -> str:
    L = ["Normal-ECG morphology checks (validate the forward geometry)",
         "-" * 62]
    for desc, ok, detail in checks:
        L.append(f"  [{'PASS' if ok else 'FAIL'}] {desc:40s} {detail}")
    n_ok = sum(1 for _, ok, _ in checks if ok)
    L.append(f"  {n_ok} of {len(checks)} checks pass")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Bounded-torso forward solution
# ---------------------------------------------------------------------------

def _source_bins(beat):
    """Compress the source into (template, activation-bin) weight coordinates.

    Every node's voltage trace is one of ``n_templates`` shapes shifted by that
    node's activation bin, so a node is fully described by the pair
    ``(template, bin)``. There are at most ``n_templates * n_bins`` such pairs
    (a few thousand), against 355k nodes, and any lead-field-weighted sum over
    nodes collapses onto those pairs. That compression is what makes a surface
    of several hundred observation points affordable.
    """
    bank = beat.bank
    at = beat.activation.activation_time
    reach = np.isfinite(at)
    nt = beat.t.size

    at_idx = np.zeros(at.size, dtype=np.int64)
    at_idx[reach] = np.round((at[reach] - at[reach].min()) / bank.dt).astype(np.int64)
    at_idx = np.clip(at_idx, 0, nt - 1)

    group = bank.index_for(beat.mesh.rho.astype(float))
    n_bins = int(at_idx[reach].max()) + 1
    key = group * n_bins + at_idx
    return reach, key, n_bins, group


def surface_potentials_unbounded(beat, surface_points_cm: np.ndarray,
                                 cond: Conductivity | None = None,
                                 chunk_points: int = 64) -> np.ndarray:
    """Unbounded-medium potential time series at arbitrary observation points.

    Returns ``(n_points, n_time)`` in mV. Observation points are processed in
    chunks so the intermediate weight matrix stays bounded: the full matrix for
    a 600-vertex torso would be about 1.7 GB.

    Within a chunk the node weights are compressed onto (template, activation
    bin) pairs by a sparse scatter, then each template's bin-weight row is
    convolved with that template by FFT. Direct accumulation over shifts would
    be roughly ``n_points * n_templates * n_bins`` array operations, which is
    far too slow once ``n_points`` leaves single digits.
    """
    from scipy.sparse import csr_matrix
    from scipy.signal import fftconvolve

    cond = cond or Conductivity()
    bank = beat.bank
    mesh = beat.mesh
    nt = beat.t.size
    rest = float(bank.Vm[:, 0].mean())

    reach, key, n_bins, _ = _source_bins(beat)
    n_g = bank.n_templates
    n_keys = n_g * n_bins

    idx = np.flatnonzero(reach)
    scatter = csr_matrix(
        (np.ones(idx.size), (key[idx], idx)),
        shape=(n_keys, mesh.n_nodes))

    pts = np.asarray(surface_points_cm, dtype=float)
    out = np.zeros((pts.shape[0], nt))

    for lo in range(0, pts.shape[0], chunk_points):
        hi = min(lo + chunk_points, pts.shape[0])
        w = lead_field_weights(mesh, pts[lo:hi], cond=cond)      # (m, n_nodes)

        W = (scatter @ w.T).T                                     # (m, n_keys)
        acc = np.zeros((hi - lo, nt))
        for g in range(n_g):
            Wg = W[:, g * n_bins:(g + 1) * n_bins]
            if not Wg.any():
                continue
            conv = fftconvolve(Wg, (bank.Vm[g] - rest)[None, :],
                               mode="full", axes=1)
            acc += conv[:, :nt]

        # Nodes still at rest contribute a constant through their weights.
        acc += rest * w[:, reach].sum(axis=1)[:, None]
        out[lo:hi] = acc
    return out


def nearest_surface_vertex(surface, positions_body: dict) -> dict:
    """Snap electrode sites onto the torso surface, reporting the shift."""
    out = {}
    for name, xyz in positions_body.items():
        d = np.linalg.norm(surface.verts - np.asarray(xyz, dtype=float), axis=1)
        k = int(np.argmin(d))
        out[name] = (k, float(d[k]))
    return out


@dataclass
class BemResult:
    """Bounded-torso forward solution for one beat."""

    frame: AnatomicalFrame
    surface: object
    leads: dict
    electrode_vertices: dict
    phi_surface: np.ndarray          # (n_verts, n_time) mV, zero-mean
    phi_unbounded: np.ndarray        # (n_verts, n_time) mV, before the boundary
    conductivity: Conductivity

    def amplification(self) -> float:
        """How much the torso boundary raised surface potentials overall."""
        a = np.linalg.norm(self.phi_surface)
        b = np.linalg.norm(self.phi_unbounded)
        return float(a / b) if b > 0 else float("nan")

    def summary(self) -> str:
        L = ["Bounded-torso forward solution (boundary element)",
             "-" * 62,
             f"  surface vertices / triangles : "
             f"{self.surface.n_verts} / {self.surface.n_tris}",
             f"  enclosed volume              : "
             f"{self.surface.enclosed_volume() / 1000.0:.1f} L",
             f"  boundary amplification       : {self.amplification():.2f}x "
             f"vs unbounded",
             "  electrode snap distances (cm to nearest surface vertex):"]
        for name, (k, d) in self.electrode_vertices.items():
            L.append(f"    {name:3s} {d:5.2f}")
        return "\n".join(L)


def run_12lead_bem(beat, cond: Conductivity | None = None,
                   frame: AnatomicalFrame | None = None,
                   surface=None, n_theta: int = 24,
                   chunk_points: int = 64) -> BemResult:
    """Twelve leads from a bounded, homogeneous torso.

    Pipeline: build the torso surface in body coordinates, map it into mesh
    coordinates through the validated anatomical frame, evaluate the unbounded
    potential at every surface vertex, solve the boundary integral equation for
    the true surface potentials, then read the electrodes off the surface.

    The boundary is what the unbounded model was missing. Because an insulated
    body cannot shed current, the surface potential rises well above the
    free-space value; for a centred dipole in a sphere the factor is exactly 3
    (see :func:`ecg12sim.torso.validate_sphere`).
    """
    from . import torso as torso_mod

    cond = cond or Conductivity()
    frame = frame or build_frame(beat.mesh)
    surface = surface if surface is not None else torso_mod.make_torso(
        n_theta=n_theta)

    # Surface lives in body coordinates; the source lives in mesh coordinates.
    verts_mesh = np.array([frame.to_mesh(v) for v in surface.verts])

    phi_inf = surface_potentials_unbounded(beat, verts_mesh, cond=cond,
                                           chunk_points=chunk_points)
    B = torso_mod.bem_matrix(surface)
    phi_s = torso_mod.solve_surface(B, phi_inf)

    ev = nearest_surface_vertex(surface, ELECTRODES_BODY)
    phi = {name: phi_s[k] for name, (k, _) in ev.items()}
    leads = derive_12_leads(phi)

    return BemResult(frame=frame, surface=surface, leads=leads,
                     electrode_vertices=ev, phi_surface=phi_s,
                     phi_unbounded=phi_inf, conductivity=cond)
