"""Action-potential and current templates banked by transmural depth.

This is the *reaction* half of the reaction-eikonal construction. The eikonal
solve in :mod:`ecg12sim.eikonal` says **when** each node activates; this module
says **what happens** once it does, by running the full ionic model of
:mod:`ecg12sim.ionic_model` a handful of times and reusing the traces.

A node at transmural depth ``rho`` that activates at time ``AT`` is modelled as

    Vm(node, t) = template[rho](t - AT)

so the mesh model keeps the real ionic kinetics, including explicit I_Na and
I_Kr, without integrating 355k coupled cells for 25,000 time steps.

Heterogeneity
-------------

The bank interpolates the same endo / M-cell / epi conductance scaling that
:func:`ecg12sim.cable._region_scales` imposes along the 1-D strand, but as a
*continuous* function of depth rather than three hard bands. Continuity matters
here: the T wave is written by the spread of repolarisation times across the
wall, and three discrete APDs would produce three spikes of repolarisation
instead of a smooth dispersion.

Honest limitation
-----------------

Each node's action potential is a shifted copy of a depth-indexed template, so
repolarisation is **prescribed by this bank** rather than emerging from
diffusive coupling between neighbours, as it does in the 1-D cable. The T wave
the mesh model produces is therefore a consequence of the APD gradient written
here. The QRS remains emergent, since it follows from the eikonal wavefront.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im

# Stimulus used to launch each template cell.
_STIM_AMP = 52.0        # pA/pF
_STIM_DUR = 1.0         # ms
_STIM_DELAY = 0.0       # ms, templates start at activation

#: Quiescent time integrated before the stimulus, letting the gates relax to
#: their true resting steady state.
#:
#: :func:`ecg12sim.ionic_model.initial_state` starts ``xr2`` at 1.0, but its
#: resting steady state at -86 mV is 0.481; every other gate starts within
#: about 2% of equilibrium. Because ``xr2`` inactivates I_Kr and relaxes with a
#: sub-millisecond time constant, starting it at 1.0 leaves I_Kr transiently
#: un-inactivated right as the action potential overshoots, and the ~127 mV
#: driving force then produces a spurious I_Kr spike a couple of milliseconds
#: after activation. Measured on an endocardial template that artefact reached
#: 1.35 pA/pF, larger than the genuine repolarisation peak of 0.82 and so
#: large that it became the reported maximum.
#:
#: 50 ms of quiescence removes it (200 ms gives an identical result). It also
#: settles the resting potential to the model's true -87.16 mV.
_PREPACE_MS = 50.0

#: I_Kr peaks are searched from this time after activation onward.
#:
#: Even from a properly equilibrated rest state, the inactivation gate takes a
#: few milliseconds to close, so a small early I_Kr transient survives at the
#: action-potential overshoot. It is real in the model, but it is not the
#: current that writes the T wave: that is the repolarisation peak several
#: hundred milliseconds later. Reporting an I_Kr "peak" therefore means the
#: repolarisation peak, and this cutoff separates the two unambiguously.
IKR_REPOL_SEARCH_MS = 50.0


@dataclass
class Heterogeneity:
    """Smooth transmural conductance profile across the wall.

    The strand in :mod:`ecg12sim.cable` uses three hard bands. Here I_Ks is
    suppressed by a Gaussian dip centred in the M-cell region instead, so APD
    varies continuously with depth. That continuity is what lets repolarisation
    spread smoothly rather than in three discrete bursts.

    ``m_gks_dip`` carries an important calibration. Left at the strand's raw
    0.25 the isolated-cell APDs spread over 140 ms, roughly triple the
    physiological transmural dispersion, because **isolated cells exaggerate
    APD differences**: in real tissue, electrotonic coupling between neighbours
    pulls their repolarisation times together. The 1-D cable shows the coupled
    answer directly, its repolarisation window spanning about 48 ms, and the
    default here is calibrated so the bank reproduces that figure. See
    :func:`calibrate_dispersion`.
    """

    #: I_Ks scale at the centre of the M-cell region (1.0 = no suppression).
    #:
    #: Deliberately mild. A deep dip (0.25, the strand's raw value) gives the
    #: M region a 40 ms APD island, and in this model that is actively harmful:
    #: with no electrotonic coupling between neighbours to compress it, the
    #: island makes repolarisation core-last and cancels the T wave. The 1-D
    #: cable does not suffer as badly because diffusion pulls neighbouring
    #: repolarisation times together for free.
    #:
    #: So the mesh model keeps only a modest M signature on top of a monotonic
    #: endo -> epi decrease. That is an explicit trade: an exaggerated,
    #: uncoupled M-cell island against a T wave that exists at all.
    m_gks_dip: float = 0.85
    #: Transmural depth where the M-cell region is centred.
    m_rho_center: float = 0.45
    #: Gaussian width of the M-cell region in depth units.
    m_rho_width: float = 0.18
    #: I_to scale at the endocardium, ramping to 1.0 by ``ito_rho_knee``.
    endo_gto: float = 0.25
    ito_rho_knee: float = 0.35

    #: Extra I_Ks toward the epicardium, shortening epicardial APD.
    #:
    #: Without this the profile is symmetric about ``m_rho_center`` and
    #: endocardial and epicardial APD come out nearly equal (measured 308.7 vs
    #: 310.8 ms). Real epicardium repolarises distinctly *earlier* than
    #: endocardium, and that asymmetry is what makes repolarisation sweep
    #: epi -> endo and produce an upright, concordant T wave. With endo and epi
    #: equal, the T wave has no stable polarity to come from: the two halves of
    #: the wall cancel.
    epi_gks_boost: float = 1.10

    #: Transmural depth at which the epicardial boost starts ramping in.
    #:
    #: **This must stay at 0.0 for the T wave to work**, and the reason is worth
    #: recording because it is not obvious.
    #:
    #: A node repolarises at ``activation(rho) + APD(rho)``, and activation
    #: rises about 18 ms from endocardium to epicardium. So APD has to *fall*
    #: by more than 18 ms across the wall, and fall monotonically, for
    #: repolarisation to sweep epi -> endo as a directional gradient. With the
    #: knee at 0.55 the APD profile was flat across the inner half, which made
    #: ``repol(mid) = 9 + APD`` later than ``repol(endo) = 0 + APD``: the
    #: mid-wall repolarised last and the pattern became core-to-surface and
    #: symmetric rather than directional.
    #:
    #: A symmetric gradient cannot write a T wave however large it is. Measured
    #: with the knee at 0.55, both surfaces repolarised together while the core
    #: lagged 40-60 ms, the inner and outer layers each contributed negative T
    #: area (-3.57 and -2.38 mV.ms) against the core's +2.27, and the T came out
    #: inverted. Widening the epi-endo APD gap made it *worse*, not better,
    #: which is the signature of a symmetry problem rather than a magnitude one.
    epi_rho_knee: float = 0.0

    def gks(self, rho: np.ndarray | float) -> np.ndarray:
        r = np.asarray(rho, dtype=float)
        z = (r - self.m_rho_center) / max(self.m_rho_width, 1e-9)
        dip = 1.0 - (1.0 - self.m_gks_dip) * np.exp(-0.5 * z * z)

        span = max(1.0 - self.epi_rho_knee, 1e-9)
        frac = np.clip((r - self.epi_rho_knee) / span, 0.0, 1.0)
        smooth = frac * frac * (3.0 - 2.0 * frac)
        return dip * (1.0 + self.epi_gks_boost * smooth)

    def gto(self, rho: np.ndarray | float) -> np.ndarray:
        r = np.asarray(rho, dtype=float)
        frac = np.clip(r / max(self.ito_rho_knee, 1e-9), 0.0, 1.0)
        smooth = frac * frac * (3.0 - 2.0 * frac)        # smoothstep
        return self.endo_gto + (1.0 - self.endo_gto) * smooth


def gks_scale(rho: np.ndarray | float,
              het: Heterogeneity | None = None) -> np.ndarray:
    """I_Ks conductance scale as a continuous function of transmural depth."""
    return (het or Heterogeneity()).gks(rho)


def gto_scale(rho: np.ndarray | float,
              het: Heterogeneity | None = None) -> np.ndarray:
    """I_to conductance scale as a continuous function of transmural depth."""
    return (het or Heterogeneity()).gto(rho)


@dataclass
class TemplateBank:
    """Action potential and current traces for a ladder of transmural depths.

    All trace arrays have shape ``(n_templates, nt)`` on the common time base
    ``t`` measured from each cell's own activation.
    """

    t: np.ndarray             # (nt,) ms since activation
    rho: np.ndarray           # (n_templates,) representative depth
    Vm: np.ndarray            # (n_templates, nt) mV
    I_Na: np.ndarray          # (n_templates, nt) pA/pF
    I_Kr: np.ndarray          # (n_templates, nt) pA/pF
    apd90: np.ndarray         # (n_templates,) ms
    repol_time: np.ndarray    # (n_templates,) ms after activation, -40 mV crossing
    dt: float                 # ms, template sample spacing

    @property
    def n_templates(self) -> int:
        return int(self.rho.size)

    def index_for(self, rho: np.ndarray) -> np.ndarray:
        """Map node depths onto template indices (nearest depth)."""
        edges = (self.rho[:-1] + self.rho[1:]) / 2.0
        return np.searchsorted(edges, np.asarray(rho, dtype=float))

    def region(self) -> np.ndarray:
        """endo / mid / epi code per template, for per-region reporting."""
        from .mesh import region_from_rho
        return region_from_rho(self.rho)

    def peak_index(self, field: np.ndarray, inward: bool,
                   t_min: float = 0.0) -> np.ndarray:
        """Index of each template's extreme, searching from ``t_min`` onward.

        ``t_min`` exists to skip the early I_Kr transient; see
        :data:`IKR_REPOL_SEARCH_MS`.
        """
        start = int(np.searchsorted(self.t, t_min))
        start = min(start, field.shape[1] - 1)
        sub = field[:, start:]
        idx = sub.argmin(axis=1) if inward else sub.argmax(axis=1)
        return idx + start

    def ikr_repol_peak(self) -> tuple[np.ndarray, np.ndarray]:
        """Each template's repolarisation I_Kr peak and its time."""
        idx = self.peak_index(self.I_Kr, inward=False,
                              t_min=IKR_REPOL_SEARCH_MS)
        rows = np.arange(self.n_templates)
        return self.I_Kr[rows, idx], self.t[idx]

    def summary(self) -> str:
        ikr_peak, ikr_t = self.ikr_repol_peak()
        L = [
            f"Template bank: {self.n_templates} depths, "
            f"{self.t[-1]:.0f} ms at dt={self.dt:g} ms",
            "I_Kr peak is the repolarisation peak (early transient excluded).",
            f"{'rho':>6s} {'APD90':>8s} {'repol':>8s} "
            f"{'I_Na peak':>10s} {'I_Kr peak':>10s} {'t_peak':>8s}",
        ]
        for i in range(self.n_templates):
            L.append(f"{self.rho[i]:6.2f} {self.apd90[i]:8.1f} "
                     f"{self.repol_time[i]:8.1f} "
                     f"{self.I_Na[i].min():10.1f} "
                     f"{ikr_peak[i]:10.3f} {ikr_t[i]:8.1f}")
        return "\n".join(L)


def _run_ensemble(rho: np.ndarray, drug: im.Drug, cond: im.Conductances,
                  duration: float, dt: float, het: Heterogeneity,
                  prepace: float = _PREPACE_MS):
    """Integrate every template depth simultaneously as one ensemble.

    :func:`ecg12sim.ionic_model.step` is already vectorised over cells, so the
    whole ladder of depths advances in one call per time step. Integrating them
    one at a time instead costs roughly ``n_templates`` times as much for an
    identical result.
    """
    n_cells = int(rho.size)
    n = int(round(duration / dt))
    gks = het.gks(rho)
    gto = het.gto(rho)

    s = im.initial_state(n_cells)

    # Relax to the true resting steady state before stimulating. See _PREPACE_MS.
    zero_pre = np.zeros(n_cells)
    for _ in range(int(round(prepace / dt))):
        s, _ = im.step(s, dt, zero_pre, cond, drug,
                       gto_scale=gto, gks_scale=gks)

    V = np.zeros((n, n_cells))
    ina = np.zeros((n, n_cells))
    ikr = np.zeros((n, n_cells))

    stim_vec = np.full(n_cells, _STIM_AMP)
    zero_vec = np.zeros(n_cells)

    for k in range(n):
        t = k * dt
        on = _STIM_DELAY <= t < _STIM_DELAY + _STIM_DUR
        s, cur = im.step(s, dt, stim_vec if on else zero_vec, cond, drug,
                         gto_scale=gto, gks_scale=gks)
        V[k] = s[0]
        ina[k] = cur["I_Na"]
        ikr[k] = cur["I_Kr"]

    return V, ina, ikr


def _apd90_and_repol(V: np.ndarray, dt: float) -> tuple[float, float]:
    """APD90 and the -40 mV downstroke crossing, both ms after activation."""
    vrest = float(V[0])
    peak = float(V.max())
    if peak <= vrest:
        return float("nan"), float("nan")

    v90 = peak - 0.9 * (peak - vrest)
    up = int(np.argmax(V > -40.0))
    below = np.flatnonzero(V[up:] < v90)
    apd90 = float(below[0] * dt) if below.size else float("nan")

    down = np.flatnonzero((V[:-1] >= -40.0) & (V[1:] < -40.0))
    repol = float(down[0] * dt) if down.size else float("nan")
    return apd90, repol


def build_template_bank(drug: im.Drug | None = None,
                        cond: im.Conductances | None = None,
                        n_templates: int = 15,
                        duration: float = 500.0,
                        dt: float = 0.02,
                        out_dt: float = 0.1,
                        het: Heterogeneity | None = None,
                        prepace: float = _PREPACE_MS) -> TemplateBank:
    """Run the ionic model across a ladder of transmural depths.

    ``dt`` integrates the ionic model; ``out_dt`` is the grid the templates are
    resampled onto. Resampling uses a **peak-preserving** reduction rather than
    plain subsampling, because I_Na is a roughly 1 ms spike: taking every Nth
    sample would under-report its amplitude by an amount that depends on where
    the samples happen to land.
    """
    drug = drug or im.Drug()
    cond = cond or im.Conductances()
    het = het or Heterogeneity()

    rho = np.linspace(0.0, 1.0, n_templates)
    stride = max(1, int(round(out_dt / dt)))
    n_full = int(round(duration / dt))
    n_out = n_full // stride
    usable = n_out * stride

    V, ina, ikr = _run_ensemble(rho, drug, cond, duration, dt, het,
                                prepace=prepace)

    apd = np.zeros(n_templates)
    rep = np.zeros(n_templates)
    for i in range(n_templates):
        apd[i], rep[i] = _apd90_and_repol(V[:, i], dt)

    # Reshape to (n_out, stride, n_templates) and reduce along the bin axis.
    # Voltage is smooth, so plain decimation is faithful. The currents keep
    # their extreme within each bin instead: I_Na is a roughly 1 ms inward
    # spike, and taking every Nth sample would under-report its amplitude by
    # an amount depending on where the samples happen to land.
    Vm = V[:usable].reshape(n_out, stride, n_templates)[:, 0, :].T
    I_Na = ina[:usable].reshape(n_out, stride, n_templates).min(axis=1).T
    I_Kr = ikr[:usable].reshape(n_out, stride, n_templates).max(axis=1).T

    return TemplateBank(t=np.arange(n_out) * (stride * dt), rho=rho,
                        Vm=np.ascontiguousarray(Vm),
                        I_Na=np.ascontiguousarray(I_Na),
                        I_Kr=np.ascontiguousarray(I_Kr),
                        apd90=apd, repol_time=rep, dt=stride * dt)


#: Coupled transmural repolarisation dispersion measured from the 1-D cable,
#: used as the calibration target for the template bank. See
#: :func:`ecg12sim.tracking.ecg_windows`.
CABLE_DISPERSION_MS = 47.8


def dispersion_for(m_gks_dip: float, n_templates: int = 15,
                   duration: float = 600.0, dt: float = 0.02) -> float:
    """APD90 dispersion across the wall for a given M-cell I_Ks suppression."""
    het = Heterogeneity(m_gks_dip=m_gks_dip)
    bank = build_template_bank(n_templates=n_templates, duration=duration,
                               dt=dt, het=het)
    apd = bank.apd90[np.isfinite(bank.apd90)]
    return float(apd.max() - apd.min()) if apd.size else float("nan")


def calibrate_dispersion(target_ms: float = CABLE_DISPERSION_MS,
                         bounds: tuple[float, float] = (0.25, 0.98),
                         tol: float = 1.5, max_iter: int = 12) -> float:
    """Find the ``m_gks_dip`` whose isolated-cell APD spread matches ``target_ms``.

    Suppressing I_Ks more deeply lengthens the M-cell action potential, so
    dispersion falls monotonically as the dip rises towards 1.0, which makes a
    bisection safe. The target defaults to the dispersion the coupled 1-D cable
    actually produces, so the mesh model inherits a tissue-level figure rather
    than the inflated isolated-cell one.
    """
    lo, hi = bounds
    d_lo = dispersion_for(lo)
    d_hi = dispersion_for(hi)
    if not (np.isfinite(d_lo) and np.isfinite(d_hi)):
        return Heterogeneity().m_gks_dip
    if target_ms >= d_lo:
        return lo
    if target_ms <= d_hi:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = dispersion_for(mid)
        if not np.isfinite(d_mid):
            break
        if abs(d_mid - target_ms) <= tol:
            return float(mid)
        if d_mid > target_ms:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
