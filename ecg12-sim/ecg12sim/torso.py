"""Bounded homogeneous volume conductor, solved by boundary elements.

Why this exists
---------------

:mod:`ecg12sim.forward` places electrodes in an *unbounded* medium. That is
fatal for the limb leads: a dipole field falls as 1/r^2, so electrodes 28-40 cm
away read about 4% of the precordial amplitude, where a real ECG has them
comparable. The cause is physical rather than numerical. A real torso is a
bounded conductor surrounded by insulating air, so no current escapes; the
current is forced to circulate within the body, which raises the surface
potential far above the free-space value.

How much it matters is exactly calculable in one case. For a current dipole at
the centre of a homogeneous sphere of radius ``R`` with an insulating boundary,
the surface potential is

    phi = 3 (p . rhat) / (4 pi sigma R^2)

while the same dipole in an unbounded medium gives, at that radius,

    phi_inf = (p . rhat) / (4 pi sigma R^2)

So the boundary amplifies by **exactly three**. That factor is the analytic test
this module is validated against (:func:`validate_sphere`), and it is also a
direct measure of how much the unbounded model was throwing away.

Method
------

For a homogeneous conductor with an insulating boundary, Green's identity turns
the problem into an integral equation over the surface alone. Discretising with
a potential value per vertex gives

    phi_S = phi_inf_S + B phi_S      =>      (I - B) phi_S = phi_inf_S

where ``B`` holds the solid angles each surface patch subtends at each surface
point. Solid angles are evaluated exactly with the Van Oosterom & Strackee
formula rather than approximated, which keeps the matrix accurate for the
near-tangential patches that dominate.

The operator is singular because potential is only defined up to a constant
(nothing grounds an insulated body), so the system is deflated by pinning the
surface mean to zero, which is also the physical reference an ECG uses.

Scope
-----

The torso is **homogeneous**: one conductivity, no separate lung, blood or
skeletal-muscle regions. The surface is parametric adult anatomy, not a
segmented subject. What it adds over :mod:`ecg12sim.forward` is the boundary
itself, which is the dominant correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Surface:
    """Closed triangulated surface."""

    verts: np.ndarray          # (n_v, 3)
    tris: np.ndarray           # (n_t, 3) int

    @property
    def n_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def n_tris(self) -> int:
        return int(self.tris.shape[0])

    def tri_normals_areas(self):
        """Outward normals (unit) and areas for every triangle."""
        p = self.verts[self.tris]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        two_a = np.linalg.norm(n, axis=1)
        return n / np.maximum(two_a, 1e-30)[:, None], two_a / 2.0

    def enclosed_volume(self) -> float:
        """Signed volume via the divergence theorem; positive if outward-oriented."""
        p = self.verts[self.tris]
        return float(np.einsum("ij,ij->i",
                               p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)

    def centroid(self) -> np.ndarray:
        return self.verts.mean(axis=0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _ring_surface(z: np.ndarray, half_width: np.ndarray, half_depth: np.ndarray,
                  n_theta: int = 24) -> Surface:
    """Build a closed surface from stacked elliptical rings, capped at both ends.

    Triangles are oriented so that normals point outward, which the solid-angle
    integral and the enclosed-volume check both depend on.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)

    rings = []
    for zi, a, b in zip(z, half_width, half_depth):
        rings.append(np.column_stack([a * ct, b * st, np.full(n_theta, zi)]))
    verts = np.vstack(rings)

    n_z = len(z)
    tris = []
    for k in range(n_z - 1):
        for i in range(n_theta):
            j = (i + 1) % n_theta
            v00 = k * n_theta + i
            v01 = k * n_theta + j
            v10 = (k + 1) * n_theta + i
            v11 = (k + 1) * n_theta + j
            # Outward orientation for a +z-stacked, counter-clockwise ring.
            tris.append((v00, v01, v11))
            tris.append((v00, v11, v10))

    bottom = verts.shape[0]
    top = bottom + 1
    verts = np.vstack([verts,
                       [0.0, 0.0, z[0]],
                       [0.0, 0.0, z[-1]]])
    for i in range(n_theta):
        j = (i + 1) % n_theta
        tris.append((bottom, j, i))                       # bottom cap, outward -z
        last = (n_z - 1) * n_theta
        tris.append((top, last + i, last + j))            # top cap, outward +z

    return Surface(verts=verts, tris=np.array(tris, dtype=np.int64))


def make_sphere(radius: float = 10.0, n_theta: int = 24,
                n_z: int = 24) -> Surface:
    """Sphere built from the same ring machinery, for the analytic test."""
    # Rings placed at equal polar-angle steps, excluding the poles.
    phi = np.linspace(0.0, np.pi, n_z + 2)[1:-1]
    z = radius * np.cos(phi)
    r = radius * np.sin(phi)
    s = _ring_surface(z[::-1], r[::-1], r[::-1], n_theta=n_theta)
    # Replace the flat caps with the true poles so the shape closes smoothly.
    s.verts[-2] = [0.0, 0.0, -radius]
    s.verts[-1] = [0.0, 0.0, radius]
    return s


#: Adult thorax cross-sections, centimetres, in the body frame used by
#: :mod:`ecg12sim.forward` (x left, y anterior, z superior, origin in the
#: mid-sagittal plane at heart level). Half-width is the left-right semi-axis,
#: half-depth the anterior-posterior one.
TORSO_PROFILE = (
    # (z, half_width, half_depth)
    (-42.0, 2.0, 2.0),      # closed inferior end, below the hips
    (-36.0, 15.0, 11.5),    # hips
    (-28.0, 14.5, 11.0),
    (-20.0, 13.5, 10.5),    # waist
    (-12.0, 14.5, 11.0),
    (-4.0, 15.0, 11.0),
    (0.0, 15.0, 11.0),      # heart level
    (6.0, 15.5, 11.0),
    (12.0, 16.5, 11.0),
    (18.0, 18.0, 11.0),     # upper chest
    (24.0, 18.5, 10.5),     # shoulders
    (29.0, 8.0, 7.0),       # neck taper
    (32.0, 2.0, 2.0),       # closed superior end
)


def make_torso(n_theta: int = 24, profile=TORSO_PROFILE,
               refine_z: int = 2) -> Surface:
    """Parametric adult torso surface in body coordinates, centimetres."""
    prof = np.array(profile, dtype=float)
    z, a, b = prof[:, 0], prof[:, 1], prof[:, 2]

    if refine_z > 1:
        zf = []
        for k in range(len(z) - 1):
            zf.append(np.linspace(z[k], z[k + 1], refine_z + 1)[:-1])
        zf.append([z[-1]])
        z_new = np.concatenate(zf)
        a = np.interp(z_new, z, a)
        b = np.interp(z_new, z, b)
        z = z_new

    return _ring_surface(z, a, b, n_theta=n_theta)


# ---------------------------------------------------------------------------
# Boundary element operator
# ---------------------------------------------------------------------------

def _solid_angles(points: np.ndarray, verts: np.ndarray,
                  tris: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Solid angle each triangle subtends at each point, shape (n_pts, n_tris).

    Uses the exact closed form of Van Oosterom & Strackee (1983):

        tan(omega / 2) = R1 . (R2 x R3)
                         / (|R1||R2||R3| + (R1.R2)|R3|
                            + (R1.R3)|R2| + (R2.R3)|R1|)

    Exact evaluation matters here: the patches closest to an observation point
    subtend most of the angle, and that is precisely where a far-field or
    centroid approximation is worst.
    """
    n_pts = points.shape[0]
    out = np.empty((n_pts, tris.shape[0]))
    p1 = verts[tris[:, 0]]
    p2 = verts[tris[:, 1]]
    p3 = verts[tris[:, 2]]

    for lo in range(0, n_pts, chunk):
        hi = min(lo + chunk, n_pts)
        o = points[lo:hi][:, None, :]                      # (m, 1, 3)
        r1 = p1[None, :, :] - o
        r2 = p2[None, :, :] - o
        r3 = p3[None, :, :] - o
        n1 = np.linalg.norm(r1, axis=2)
        n2 = np.linalg.norm(r2, axis=2)
        n3 = np.linalg.norm(r3, axis=2)

        num = np.einsum("mtj,mtj->mt", r1, np.cross(r2, r3))
        den = (n1 * n2 * n3
               + np.einsum("mtj,mtj->mt", r1, r2) * n3
               + np.einsum("mtj,mtj->mt", r1, r3) * n2
               + np.einsum("mtj,mtj->mt", r2, r3) * n1)
        out[lo:hi] = 2.0 * np.arctan2(num, den)
    return out


def bem_matrix(surface: Surface) -> np.ndarray:
    """Assemble the deflated BEM operator ``B`` with ``(I - B) phi = phi_inf``.

    Each triangle's solid angle at an observation vertex is shared equally among
    its three vertices, which is the standard linear-collocation distribution.
    The diagonal is then set by the requirement that a constant potential
    produces no field: rows of ``B`` must sum to one for an interior point, so
    the self term is whatever is left over. Imposing that identity exactly is
    more robust than evaluating the singular self-patch analytically, and it
    guarantees the discrete operator preserves constants.
    """
    verts = surface.verts
    tris = surface.tris
    n_v = surface.n_verts

    omega = _solid_angles(verts, verts, tris)              # (n_v, n_t)

    # Scatter each triangle's angle to its three vertices.
    B = np.zeros((n_v, n_v))
    share = omega / (4.0 * np.pi) / 3.0
    for c in range(3):
        np.add.at(B.T, tris[:, c], share.T)

    # Enforce exact preservation of constants: sum_j B_ij = 1.
    np.fill_diagonal(B, 0.0)
    np.fill_diagonal(B, 1.0 - B.sum(axis=1))
    return B


def solve_surface(B: np.ndarray, phi_inf: np.ndarray) -> np.ndarray:
    """Solve ``(I - B) phi = phi_inf`` for surface potentials.

    ``I - B`` is singular by construction: an insulated body has no ground, so
    the potential is fixed only up to an additive constant. The system is
    deflated by pinning the surface mean to zero, which is also the reference an
    ECG implicitly uses. ``phi_inf`` may be ``(n_v,)`` or ``(n_v, n_time)``.
    """
    n = B.shape[0]
    A = np.eye(n) - B
    # Deflation: replace the null direction with the zero-mean constraint.
    A = A + np.full((n, n), 1.0 / n)
    rhs = np.atleast_2d(phi_inf.T).T
    rhs = rhs - rhs.mean(axis=0, keepdims=True)
    sol = np.linalg.solve(A, rhs)
    sol = sol - sol.mean(axis=0, keepdims=True)
    return sol[:, 0] if phi_inf.ndim == 1 else sol


# ---------------------------------------------------------------------------
# Analytic validation
# ---------------------------------------------------------------------------

def validate_sphere(radius: float = 10.0, n_theta: int = 32,
                    n_z: int = 32, sigma: float = 0.2) -> dict:
    """Check the BEM against the centred-dipole-in-a-sphere solution.

    A dipole at the centre of a homogeneous insulating sphere gives surface
    potentials exactly three times the unbounded-medium value at the same
    radius. Recovering 3.0 validates the operator's constant, its sign and the
    solid-angle assembly all at once.
    """
    s = make_sphere(radius, n_theta=n_theta, n_z=n_z)
    p = np.array([0.0, 0.0, 1.0])                          # unit dipole, +z

    r = s.verts
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    # Unbounded-medium potential of a dipole at the origin.
    phi_inf = (r @ p) / (4.0 * np.pi * sigma * rn[:, 0] ** 3)

    phi = solve_surface(bem_matrix(s), phi_inf)

    analytic = 3.0 * (r @ p) / (4.0 * np.pi * sigma * radius ** 3)
    # Both are zero-mean by symmetry; compare by least-squares gain.
    gain = float(np.dot(phi, analytic) / np.dot(analytic, analytic))
    resid = float(np.linalg.norm(phi - analytic) / np.linalg.norm(analytic))
    inf_gain = float(np.dot(phi, phi_inf) / np.dot(phi_inf, phi_inf))

    return {
        "n_verts": s.n_verts,
        "n_tris": s.n_tris,
        "volume_ratio": s.enclosed_volume() / (4.0 / 3.0 * np.pi * radius ** 3),
        "amplification_vs_unbounded": inf_gain,
        "gain_vs_analytic": gain,
        "relative_residual": resid,
    }


def format_sphere_check(res: dict) -> str:
    L = ["BEM validation: dipole at the centre of an insulating sphere",
         "-" * 62,
         f"  surface vertices / triangles : {res['n_verts']} / {res['n_tris']}",
         f"  enclosed volume / exact      : {res['volume_ratio']:.4f}",
         f"  amplification vs unbounded   : "
         f"{res['amplification_vs_unbounded']:.4f}  (exact: 3.0)",
         f"  gain vs analytic solution    : {res['gain_vs_analytic']:.4f} "
         f" (exact: 1.0)",
         f"  relative residual            : {res['relative_residual']:.4f}"]
    return "\n".join(L)
