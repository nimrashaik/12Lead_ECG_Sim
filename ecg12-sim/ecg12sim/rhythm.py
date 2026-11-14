"""Multi-beat rhythm: a full 12-lead strip at a prescribed heart rate.

Everything upstream of this module computes **one** beat. A clinical ECG is a
ten-second recording, so this module turns single-beat solutions into a rhythm
strip, and it is also where the atria and the ventricles are finally combined
into one signal.

Two sources, one torso
----------------------

The atrial and ventricular meshes come from the same cohort file and therefore
share a coordinate frame, so a single anatomical frame, a single torso surface
and a single set of electrodes see both. The forward problem is **linear** in the
source, which is what makes the combination exact rather than approximate: the
lead field is a fixed matrix of geometry and conductivity, and the boundary
element operator is a fixed matrix too. So the potential due to atria and
ventricles together is the sum of the potentials due to each, and

    phi_electrode(t) = phi_atrial(t) + phi_ventricular(t - PR)

is not a modelling shortcut but the actual solution.

Why the atria are offset rather than coupled
--------------------------------------------

The atrial and ventricular tag sets share 5,480 nodes around the valve annuli
(see :mod:`ecg12sim.atria`), so they are solved as separate meshes. The
ventricular beat is then simply started ``pr_interval_ms`` after the atrial
beat. That offset *is* the PR interval, and it stands in for the whole
atrioventricular pathway: the AV node's decremental conduction, the His bundle
and the bundle branches, none of which exist as tissue in this mesh. Prescribing
it is honest about that; it is one number, it is clinically measurable, and the
default of 160 ms sits mid-range in the normal 120-200 ms window.

Why beats superpose instead of being re-simulated
-------------------------------------------------

Each beat is the same single-beat solution shifted in time, so a rhythm is a sum
of shifted copies. This is exact for the forward problem, and it is what makes a
ten-second twelve-lead strip cost the same as two single beats rather than
twelve. It does, however, assume **every beat is identical**, and that is a real
limitation worth naming: a real heart's action-potential duration shortens as
rate rises (restitution), and the templates here are built from a single
stimulus rather than from steady-state pacing at the target rate. At 72 bpm the
effect is small, because the 833 ms cycle leaves the ventricles a long stretch
of electrical silence after repolarisation, so successive beats do not interact
at all. At high rates they would: see
:meth:`RhythmConfig.diastolic_margin_ms`, which measures the remaining margin
from the assembled signal and is reported so the assumption can be checked
rather than trusted.

Where beats *do* overlap, the superposition still handles it correctly as far as
the forward model is concerned: the overlap of one beat's T wave with the next
beat's P wave is summed, which is what a real electrocardiogram shows too. What
superposition cannot capture is the overlap changing the beats themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import forward as fwd
from . import torso as torso_mod

#: Normal clinical ranges, used by the morphology checks.
P_DURATION_MS = (80.0, 120.0)
P_AMPLITUDE_MV = 0.25           # upper limit in lead II
PR_INTERVAL_MS = (120.0, 200.0)
P_AXIS_DEG = (0.0, 75.0)


@dataclass
class RhythmConfig:
    """Timing of the rhythm strip."""

    heart_rate_bpm: float = 72.0
    duration_s: float = 10.0

    #: Atrioventricular delay, ms: the offset from the start of **atrial**
    #: activation (the sinoatrial node firing) to the start of **ventricular**
    #: activation (the earliest Purkinje breakthrough). It stands in for the AV
    #: node, the His bundle and the bundle branches, none of which exist as
    #: tissue in this mesh.
    #:
    #: It is not identical to a PR interval read off the finished strip, because
    #: a measured PR starts at the first *detectable* P deflection rather than
    #: at the true first instant of atrial activation. Measured on the assembled
    #: strip, that difference turns out to be small: with the 160 ms default,
    #: :func:`measure_beats` recovers a P onset 2.8 ms after the simulated one
    #: and a PR interval of 158 ms. :func:`check_p_wave` reports the measured
    #: value against the clinical 120-200 ms window and names this prescribed
    #: value beside it, so the two can be compared rather than conflated.
    pr_interval_ms: float = 160.0

    @property
    def rr_ms(self) -> float:
        if self.heart_rate_bpm <= 0:
            raise ValueError("heart rate must be positive")
        return 60_000.0 / self.heart_rate_bpm

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0

    @property
    def n_beats(self) -> int:
        """Beats whose atrial onset falls inside the strip."""
        return int(np.floor((self.duration_ms - 1e-9) / self.rr_ms)) + 1

    def atrial_onsets_ms(self) -> np.ndarray:
        return np.arange(self.n_beats, dtype=float) * self.rr_ms

    def ventricular_onsets_ms(self) -> np.ndarray:
        return self.atrial_onsets_ms() + self.pr_interval_ms

    def diastolic_margin_ms(self, vent_beat_ms: float) -> float:
        """Quiet time between the end of one beat and the next beat's P onset.

        Negative means beats overlap, so the identical-beat assumption that
        superposition rests on is no longer safe. ``vent_beat_ms`` should be the
        duration over which the ventricular beat is *electrically active*, not
        the length of its template array: the template runs 600 ms but the
        signal has decayed by around 450 ms, and using the array length would
        understate the margin by the length of the flat tail.
        """
        return self.rr_ms - self.pr_interval_ms - vent_beat_ms

    def summary(self) -> str:
        return "\n".join([
            "Rhythm",
            f"  heart rate      : {self.heart_rate_bpm:.1f} bpm "
            f"(RR {self.rr_ms:.1f} ms)",
            f"  strip duration  : {self.duration_s:.1f} s",
            f"  beats in strip  : {self.n_beats}",
            f"  PR interval     : {self.pr_interval_ms:.0f} ms "
            f"(prescribed: stands in for AV node and His-Purkinje)",
        ])


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def superpose(single: np.ndarray, onsets_ms: np.ndarray, dt: float,
              nt_total: int) -> np.ndarray:
    """Accumulate shifted copies of a single-beat signal onto a rhythm grid.

    ``single`` is ``(n_channels, nt_beat)`` and must be a *deviation* signal:
    zero before activation and decayed back to zero at its end, so that copies
    add without stepping the baseline. Copies are clipped at the end of the
    strip, which is what a recording that stops mid-beat looks like.
    """
    single = np.atleast_2d(single)
    out = np.zeros((single.shape[0], nt_total))
    for o in np.atleast_1d(onsets_ms):
        s = int(round(float(o) / dt))
        if s >= nt_total:
            continue
        if s < 0:
            seg = single[:, -s:]
            n = min(seg.shape[1], nt_total)
            out[:, :n] += seg[:, :n]
            continue
        n = min(single.shape[1], nt_total - s)
        if n > 0:
            out[:, s:s + n] += single[:, :n]
    return out


def _active_duration(single: np.ndarray, dt: float,
                     frac: float = 0.01) -> float:
    """How long a single-beat signal stays electrically active, ms.

    The last instant at which any channel still exceeds ``frac`` of the beat's
    largest deflection. Measures the beat rather than the template array, which
    carries a flat tail once repolarisation is complete.
    """
    env = np.max(np.abs(single), axis=0)
    if env.size == 0 or env.max() <= 0:
        return 0.0
    active = np.flatnonzero(env > frac * env.max())
    return float((active[-1] + 1) * dt) if active.size else 0.0


@dataclass
class RhythmResult:
    """A multi-beat 12-lead strip and the pieces it was built from."""

    t: np.ndarray                     # (nt,) ms
    leads: dict                       # name -> (nt,) mV, atria + ventricles
    leads_atrial: dict                # atrial contribution alone
    leads_ventricular: dict           # ventricular contribution alone
    config: RhythmConfig
    frame: object
    surface: object
    electrode_vertices: dict
    beat_leads_atrial: dict           # single atrial beat, for detail plots
    beat_leads_ventricular: dict
    beat_t_atrial: np.ndarray
    beat_t_ventricular: np.ndarray
    #: Atrial depolarisation span within the single atrial beat, ms. The P wave
    #: is confined to this window; beyond it the atrial trace carries the atrial
    #: T wave, whose area cancels the P wave's over a full beat, so polarity
    #: must be judged inside this window and not over the whole beat.
    p_window: tuple = (float("nan"), float("nan"))
    ta_window: tuple = (float("nan"), float("nan"))
    diastolic_margin_ms: float = float("nan")

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def summary(self) -> str:
        p2p = {n: float(np.ptp(s)) for n, s in self.leads.items()}
        pa = {n: float(np.ptp(s)) for n, s in self.leads_atrial.items()}
        L = [
            "Rhythm strip (atria + ventricles, bounded torso)",
            "-" * 62,
            f"  samples              : {self.t.size:,} at {self.dt:g} ms",
            f"  span                 : {self.t[0]:.0f} - {self.t[-1]:.0f} ms",
            f"  P wave / atrial T    : {self.p_window[0]:.0f}-"
            f"{self.p_window[1]:.0f} ms / {self.ta_window[0]:.0f}-"
            f"{self.ta_window[1]:.0f} ms after each atrial onset",
            f"  diastolic margin     : {self.diastolic_margin_ms:+.0f} ms "
            + ("(beats do not overlap)" if self.diastolic_margin_ms > 0
               else "(BEATS OVERLAP: identical-beat assumption is unsafe)"),
            "",
            f"  {'lead':<5s} {'total p2p':>10s} {'atrial p2p':>11s} "
            f"{'atrial share':>13s}",
        ]
        for n in fwd.LEAD_NAMES:
            if n in p2p:
                share = pa[n] / p2p[n] if p2p[n] > 0 else float("nan")
                L.append(f"  {n:<5s} {p2p[n]:10.4f} {pa[n]:11.4f} "
                         f"{share:12.1%}")
        return "\n".join(L)


def run_rhythm_bem(vent_beat, atrial_beat, cfg: RhythmConfig | None = None,
                   cond: fwd.Conductivity | None = None,
                   cond_atrial: fwd.Conductivity | None = None,
                   frame=None, surface=None, n_theta: int = 24,
                   chunk_points: int = 64) -> RhythmResult:
    """Twelve leads over a multi-beat strip, from atrial and ventricular sources.

    The boundary element system is assembled and factorised **once** and both
    single-beat surface solutions are obtained from it in a single
    multiple-right-hand-side solve, by concatenating them along time. That is
    exact: the deflation in :func:`ecg12sim.torso.solve_surface` acts
    column-wise, so columns from different sources do not interact.

    Only the nine electrode vertices are retained before superposition. The full
    surface over ten seconds would be 602 vertices x 100,001 samples, around
    480 MB; the electrodes are 9 x 100,001, around 7 MB.
    """
    cfg = cfg or RhythmConfig()
    cond = cond or fwd.Conductivity()
    # The atria get an isotropic intracellular conductivity, because the cohort
    # supplies no atrial fibre field to orient an anisotropic one with. See
    # forward.ATRIAL_CONDUCTIVITY.
    cond_atrial = cond_atrial or fwd.ATRIAL_CONDUCTIVITY
    frame = frame or fwd.build_frame(vent_beat.mesh)
    surface = surface if surface is not None else torso_mod.make_torso(
        n_theta=n_theta)

    dt_a = float(atrial_beat.bank.dt)
    dt_v = float(vent_beat.bank.dt)
    if abs(dt_a - dt_v) > 1e-12:
        raise ValueError(f"atrial and ventricular template banks must share a "
                         f"sample spacing (got {dt_a} and {dt_v} ms)")
    dt = dt_v

    # Both meshes live in the same file's coordinates, so one mapped surface
    # serves both sources.
    verts_mesh = np.array([frame.to_mesh(v) for v in surface.verts])

    # Deviation-only: the resting contribution is constant in time, so it only
    # sets an arbitrary baseline level. Excluding it puts diastole at 0 mV,
    # which is what an ECG displays, and keeps the superposition additive.
    phi_inf_a = fwd.surface_potentials_unbounded(
        atrial_beat, verts_mesh, cond=cond_atrial, chunk_points=chunk_points,
        include_rest=False)
    phi_inf_v = fwd.surface_potentials_unbounded(
        vent_beat, verts_mesh, cond=cond, chunk_points=chunk_points,
        include_rest=False)

    n_a = phi_inf_a.shape[1]
    B = torso_mod.bem_matrix(surface)
    phi_s = torso_mod.solve_surface(B, np.hstack([phi_inf_a, phi_inf_v]))
    phi_s_a, phi_s_v = phi_s[:, :n_a], phi_s[:, n_a:]

    ev = fwd.nearest_surface_vertex(surface, fwd.ELECTRODES_BODY)
    names = [n for n in fwd.ELECTRODE_NAMES if n in ev]
    rows = [ev[n][0] for n in names]
    beat_a = phi_s_a[rows]              # (9, nt_a)
    beat_v = phi_s_v[rows]              # (9, nt_v)

    nt = int(round(cfg.duration_ms / dt)) + 1
    t = np.arange(nt) * dt

    rhythm_a = superpose(beat_a, cfg.atrial_onsets_ms(), dt, nt)
    rhythm_v = superpose(beat_v, cfg.ventricular_onsets_ms(), dt, nt)

    def _leads(mat):
        return fwd.derive_12_leads({n: mat[i] for i, n in enumerate(names)})

    active_v = _active_duration(beat_v, dt)
    margin = cfg.diastolic_margin_ms(active_v)

    return RhythmResult(
        t=t,
        leads=_leads(rhythm_a + rhythm_v),
        leads_atrial=_leads(rhythm_a),
        leads_ventricular=_leads(rhythm_v),
        config=cfg, frame=frame, surface=surface, electrode_vertices=ev,
        beat_leads_atrial=_leads(beat_a),
        beat_leads_ventricular=_leads(beat_v),
        beat_t_atrial=np.arange(beat_a.shape[1]) * dt,
        beat_t_ventricular=np.arange(beat_v.shape[1]) * dt,
        p_window=tuple(atrial_beat.p_window),
        ta_window=tuple(atrial_beat.ta_window),
        diastolic_margin_ms=margin)


# ---------------------------------------------------------------------------
# Measuring the strip back off the waveform
# ---------------------------------------------------------------------------

@dataclass
class BeatMeasurement:
    """Intervals and amplitudes read off one beat of the finished strip."""

    r_time: float
    qrs_onset: float
    p_onset: float = float("nan")
    p_offset: float = float("nan")
    p_peak_time: float = float("nan")
    p_amplitude: float = float("nan")        # signed, in the measurement lead
    pr_interval: float = float("nan")
    p_duration: float = float("nan")

    @property
    def has_p(self) -> bool:
        return bool(np.isfinite(self.p_onset) and np.isfinite(self.p_offset))


def _lead_envelope(leads: dict) -> np.ndarray:
    """Largest absolute deflection across all leads, per sample.

    A lead-independent activity envelope. Used to find QRS complexes without
    privileging one lead, which could be isoelectric for a given axis.
    """
    return np.max(np.abs(np.stack([leads[n] for n in leads])), axis=0)


def _slope_envelope(t: np.ndarray, leads: dict,
                    smooth_ms: float = 12.0) -> np.ndarray:
    """Smoothed cross-lead maximum of ``|d/dt|``, a QRS-selective score.

    Detecting QRS complexes on *amplitude* does not work on this model's output,
    and the reason is worth stating because it is a property of the model rather
    than of the detector. The mesh model's T wave is larger than its R wave
    (measured on lead II: T peak 0.97 mV against R peak 0.52 mV), because
    ventricular repolarisation is prescribed by the template bank's APD gradient
    rather than emerging from electrotonic coupling, as
    :mod:`ecg12sim.templates` records. An amplitude threshold therefore finds
    two "beats" per cardiac cycle.

    Slope separates them cleanly, since depolarisation is fast and
    repolarisation is slow: measured here, the QRS reaches about 0.15 mV/ms
    while the T wave stays near 0.02 and the P wave near 0.005. This is the same
    reasoning behind the derivative stage of the classical Pan-Tompkins
    detector.
    """
    dt = float(t[1] - t[0])
    d = np.max(np.abs(np.stack([np.gradient(leads[n], dt) for n in leads])),
               axis=0)
    w = max(int(round(smooth_ms / dt)), 1)
    return np.convolve(d, np.ones(w) / w, mode="same")


def detect_qrs(t: np.ndarray, leads: dict,
               refractory_ms: float = 250.0,
               threshold_frac: float = 0.30) -> np.ndarray:
    """Sample indices of QRS complexes, found on the slope envelope.

    Deliberately measured from the waveform rather than taken from the
    prescribed beat times, so that comparing the two is a real check on the
    assembled strip instead of a restatement of its inputs. The returned index
    is the peak *amplitude* sample within each detected complex, which is the R
    peak, so that the reported R times are the clinically meaningful ones.
    """
    from scipy.signal import find_peaks

    slope = _slope_envelope(t, leads)
    if slope.max() <= 0:
        return np.array([], dtype=int)
    dt = float(t[1] - t[0])
    found, _ = find_peaks(slope, height=threshold_frac * slope.max(),
                          distance=max(int(refractory_ms / dt), 1))

    # Refine each detection to the largest absolute deflection nearby, so the
    # reported time is the R peak rather than the steepest point of the upstroke.
    env = _lead_envelope(leads)
    half = max(int(round(40.0 / dt)), 1)
    out = []
    for k in found:
        lo, hi = max(int(k) - half, 0), min(int(k) + half + 1, env.size)
        out.append(lo + int(np.argmax(env[lo:hi])))
    return np.array(sorted(set(out)), dtype=int)


def _qrs_onset(slope: np.ndarray, peak: int, dt: float,
               frac: float = 0.08, max_back_ms: float = 120.0) -> int:
    """Walk back from the R peak to the start of the QRS, on the slope envelope.

    Slope rather than amplitude, because the amplitude envelope has not returned
    to baseline before the QRS: the P wave is sitting there. The P wave's
    amplitude (around 0.2 mV) exceeds a few per cent of the QRS amplitude
    (around 2.9 mV in V4), so walking back on amplitude overshoots the QRS onset
    and lands somewhere in the P wave. On slope the separation is clean.
    """
    lo = max(peak - int(round(max_back_ms / dt)), 0)
    seg = slope[lo:peak + 1]
    if seg.size == 0:
        return lo
    below = np.flatnonzero(seg < frac * seg.max())
    return lo + int(below[-1]) if below.size else lo


def measure_beats(t: np.ndarray, leads: dict, p_lead: str = "II",
                  p_search_ms: float = 250.0,
                  p_threshold_frac: float = 0.05) -> list:
    """Measure P onset/offset, PR interval and P amplitude for every beat.

    The P wave is located in the interval before each QRS onset, on the
    cross-lead amplitude envelope, then its amplitude is read in ``p_lead``.
    Searching a bounded window before the QRS keeps the measurement local, so it
    does not depend on knowing the beat times; 250 ms is wide enough to contain
    a normal P wave plus PR interval and narrow enough to exclude the previous
    beat's T wave at normal rates. At high rates the previous T wave can enter
    the window, which would corrupt the P measurement rather than the strip.

    The 5% threshold defines where the P wave "leaves the baseline", making the
    measurement a clinically detectable onset rather than the true first instant
    of atrial activation. Taking the threshold on the cross-lead envelope rather
    than on a single lead keeps it sensitive: measured on the assembled strip it
    recovers a P onset 2.8 ms after the simulated one and a P duration of
    96.3 ms against a simulated atrial activation span of 94.9 ms.
    """
    dt = float(t[1] - t[0])
    env = _lead_envelope(leads)
    slope = _slope_envelope(t, leads)
    peaks = detect_qrs(t, leads)
    sig = leads[p_lead]

    out = []
    for pk in peaks:
        qrs_on = _qrs_onset(slope, int(pk), dt)
        m = BeatMeasurement(r_time=float(t[pk]), qrs_onset=float(t[qrs_on]))

        lo = max(qrs_on - int(round(p_search_ms / dt)), 0)
        hi = max(qrs_on - int(round(10.0 / dt)), lo + 1)
        if hi - lo > 2:
            win = env[lo:hi]
            rel = int(np.argmax(win))
            p_pk = lo + rel
            if win[rel] > 0:
                level = p_threshold_frac * win[rel]
                pre = np.flatnonzero(env[lo:p_pk + 1] < level)
                on = lo + int(pre[-1]) if pre.size else lo
                post = np.flatnonzero(env[p_pk:hi] < level)
                off = p_pk + int(post[0]) if post.size else hi
                m.p_onset = float(t[on])
                m.p_offset = float(t[off])
                m.p_peak_time = float(t[p_pk])
                m.p_amplitude = float(sig[p_pk])
                m.p_duration = m.p_offset - m.p_onset
                m.pr_interval = m.qrs_onset - m.p_onset
        out.append(m)
    return out


def _area(t: np.ndarray, sig: np.ndarray, lo: float, hi: float) -> float:
    m = (t >= lo) & (t <= hi)
    if m.sum() < 2:
        return 0.0
    return float(np.trapezoid(sig[m], t[m]))


def p_wave_axis_deg(t: np.ndarray, leads: dict, lo: float, hi: float) -> float:
    """Frontal-plane P axis in degrees, from lead I and aVF net areas.

    The standard construction: lead I measures the leftward component and aVF
    the inferior one, so ``atan2(aVF, I)`` is the axis in the usual clinical
    convention where 0 degrees points to the subject's left and +90 straight
    down. A normal P axis is between 0 and +75 degrees, because atrial
    activation runs from the sinoatrial node, high in the right atrium, down and
    leftward toward the atrioventricular node and the left atrium.
    """
    a_i = _area(t, leads["I"], lo, hi)
    a_f = _area(t, leads["aVF"], lo, hi)
    if a_i == 0.0 and a_f == 0.0:
        return float("nan")
    return float(np.degrees(np.arctan2(a_f, a_i)))


def check_p_wave(result: RhythmResult, p_lead: str = "II") -> list:
    """Test the strip against textbook features of normal sinus P waves.

    These are the real validation that the atrial mesh produces a *P wave*
    rather than merely some extra deflection. Polarity and axis follow from
    where the sinoatrial node sits relative to the electrodes, so they would
    fail if the atrial geometry or the anatomical frame were wrong; duration
    follows from the atrial conduction velocity; amplitude follows from the
    atrial wall being thin.

    Each entry is ``(description, passed, detail)``.
    """
    t, leads = result.t, result.leads
    cfg = result.config
    beats = measure_beats(t, leads, p_lead=p_lead)
    with_p = [b for b in beats if b.has_p]
    out = []

    out.append(("a P wave precedes every QRS",
                len(with_p) == len(beats) and len(beats) > 0,
                f"{len(with_p)} of {len(beats)} beats"))
    if not with_p:
        return out

    # Polarity, measured on the single atrial beat so the ventricular complex
    # cannot contaminate the area, and restricted to the **depolarisation**
    # window. Integrating a whole atrial beat gives ~0 by construction: the
    # atrial T wave's area cancels the P wave's, because every node returns to
    # the potential it started from. Judging P polarity over the full beat
    # therefore measures nothing at all, which is what an earlier version of
    # this check did.
    bt, bl = result.beat_t_atrial, result.beat_leads_atrial
    lo = float(result.p_window[0])
    hi = float(result.p_window[1]) + 30.0      # let the last nodes finish
    a_ii = _area(bt, bl["II"], lo, hi)
    out.append(("P upright in II", a_ii > 0, f"{a_ii:+.4f} mV.ms"))
    a_avr = _area(bt, bl["aVR"], lo, hi)
    out.append(("P inverted in aVR", a_avr < 0, f"{a_avr:+.4f} mV.ms"))
    a_i = _area(bt, bl["I"], lo, hi)
    out.append(("P upright in I", a_i > 0, f"{a_i:+.4f} mV.ms"))

    axis = p_wave_axis_deg(bt, bl, lo, hi)
    out.append((f"P axis within {P_AXIS_DEG[0]:.0f}-{P_AXIS_DEG[1]:.0f} deg",
                bool(P_AXIS_DEG[0] <= axis <= P_AXIS_DEG[1]),
                f"{axis:+.1f} deg"))

    dur = float(np.mean([b.p_duration for b in with_p]))
    out.append((f"P duration {P_DURATION_MS[0]:.0f}-{P_DURATION_MS[1]:.0f} ms",
                bool(P_DURATION_MS[0] <= dur <= P_DURATION_MS[1]),
                f"{dur:.0f} ms (mean of {len(with_p)})"))

    amp = float(np.mean([abs(b.p_amplitude) for b in with_p]))
    out.append((f"P amplitude in {p_lead} below {P_AMPLITUDE_MV:.2f} mV",
                amp <= P_AMPLITUDE_MV, f"{amp:.3f} mV"))
    out.append(("P amplitude is non-trivial (above 0.02 mV)", amp > 0.02,
                f"{amp:.3f} mV"))

    pr = float(np.mean([b.pr_interval for b in with_p]))
    out.append((f"PR interval {PR_INTERVAL_MS[0]:.0f}-"
                f"{PR_INTERVAL_MS[1]:.0f} ms",
                bool(PR_INTERVAL_MS[0] <= pr <= PR_INTERVAL_MS[1]),
                f"{pr:.0f} ms measured, {cfg.pr_interval_ms:.0f} ms prescribed "
                f"AV delay"))

    # The atrial T wave should be hidden under the ventricular complex, as it is
    # clinically. This checks it is both small and correctly timed, rather than
    # standing proud as a spurious extra deflection.
    ta_lo, ta_hi = result.ta_window
    ta_amp = float(np.max(np.abs(bl["II"][(bt >= ta_lo - 20.0)
                                          & (bt <= ta_hi + 60.0)]))) \
        if np.isfinite(ta_lo) else float("nan")
    out.append(("atrial T wave stays below the P wave in II",
                bool(ta_amp < amp),
                f"Ta {ta_amp:.3f} mV vs P {amp:.3f} mV, Ta at "
                f"{ta_lo:.0f}-{ta_hi:.0f} ms (inside the QRS-T, as clinically)"))
    return out


def check_rhythm(result: RhythmResult) -> list:
    """Test that the strip really carries the requested rate and duration."""
    t, leads, cfg = result.t, result.leads, result.config
    peaks = detect_qrs(t, leads)
    out = []

    out.append((f"strip is {cfg.duration_s:.0f} s long",
                abs(t[-1] - cfg.duration_ms) <= result.dt,
                f"{t[-1] / 1000.0:.3f} s, {t.size:,} samples"))
    out.append((f"{cfg.n_beats} QRS complexes detected",
                peaks.size == cfg.n_beats,
                f"{peaks.size} detected, {cfg.n_beats} expected"))
    if peaks.size < 2:
        return out

    rr = np.diff(t[peaks])
    out.append((f"mean RR interval is {cfg.rr_ms:.0f} ms",
                abs(rr.mean() - cfg.rr_ms) <= 2.0,
                f"{rr.mean():.1f} ms (sd {rr.std():.2f})"))
    hr = 60_000.0 / rr.mean()
    out.append((f"measured heart rate is {cfg.heart_rate_bpm:.0f} bpm",
                abs(hr - cfg.heart_rate_bpm) <= 0.5,
                f"{hr:.2f} bpm from {peaks.size} complexes"))
    out.append(("beats do not overlap at this rate",
                result.diastolic_margin_ms > 0,
                f"{result.diastolic_margin_ms:+.0f} ms of diastole left"))
    return out


def format_checks(checks: list, title: str) -> str:
    L = [title, "-" * 62]
    for desc, ok, detail in checks:
        L.append(f"  [{'PASS' if ok else 'FAIL'}] {desc:46s} {detail}")
    n_ok = sum(1 for _, ok, _ in checks if ok)
    L.append(f"  {n_ok} of {len(checks)} checks pass")
    return "\n".join(L)


def format_beat_table(result: RhythmResult, p_lead: str = "II") -> str:
    """Per-beat intervals read off the finished strip."""
    beats = measure_beats(result.t, result.leads, p_lead=p_lead)
    L = [f"Per-beat measurements read back off the strip (P in lead {p_lead})",
         "-" * 62,
         f"  {'#':>2s} {'P onset':>9s} {'P peak':>9s} {'P end':>9s} "
         f"{'P dur':>7s} {'P amp':>8s} {'QRS on':>9s} {'PR':>7s} {'R':>9s}"]
    for i, b in enumerate(beats, 1):
        L.append(f"  {i:2d} {b.p_onset:9.1f} {b.p_peak_time:9.1f} "
                 f"{b.p_offset:9.1f} {b.p_duration:7.1f} {b.p_amplitude:8.3f} "
                 f"{b.qrs_onset:9.1f} {b.pr_interval:7.1f} {b.r_time:9.1f}")
    return "\n".join(L)
