"""Synthesise the 12 standard ECG leads from the transmural simulation.

A single 1-D strand only yields one electrical axis, so it cannot by itself
reproduce the distinct morphologies of the 12 clinical leads (that genuinely
needs a 3-D torso). Instead we build a vectorcardiographic (VCG) dipole from the
simulation and map it to the 12 leads with the classic **Dower transform**
(Dower et al., 1980), a standard, citable VCG <-> 12-lead relationship.

Physiology drives the result end to end:

    * the depolarisation phase of the pseudo-ECG becomes the QRS loop,
    * the repolarisation phase becomes the T loop,
    * QRS and T are given distinct, literature-typical axis orientations so the
      synthesised leads show realistic patterns (R-wave progression V1->V6,
      negative aVR, upright T in II/V5),
    * blocking I_Na shrinks/slows the QRS in every lead, blocking I_Kr delays the
      T wave and lengthens the QT in every lead.

This 12-lead fan-out is a presentation layer; the physically grounded signal is
the single transmural pseudo-ECG in :mod:`ecg12sim.cable`.
"""

from __future__ import annotations

import numpy as np

from .cable import CableResult

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Inverse Dower coefficients: map VCG [X, Y, Z] -> 8 independent leads.
_DOWER = {
    "V1": (-0.515, 0.157, -0.917),
    "V2": (0.044, 0.164, -1.387),
    "V3": (0.882, 0.098, -1.277),
    "V4": (1.213, 0.127, -0.601),
    "V5": (1.125, 0.127, -0.086),
    "V6": (0.831, 0.076, 0.230),
    "I": (0.632, -0.235, 0.059),
    "II": (-0.235, 1.066, -0.132),
}

# Fixed axis orientations (unit-ish vectors in Dower XYZ space) chosen to give a
# textbook-normal frontal axis and precordial R-wave progression.
_QRS_AXIS = np.array([0.40, 0.70, 0.60])
_T_AXIS = np.array([0.35, 0.60, -0.10])

_T_DISPLAY_BOOST = 2.5      # fixed boost for the under-weighted 1-D T wave
_TARGET_II_R_MV = 1.2       # normalise so lead II R-wave peak ~1.2 mV


def _smoothstep(t: np.ndarray, edge: float, width: float) -> np.ndarray:
    """Smooth 0->1 ramp centred on ``edge`` with the given transition ``width``."""
    z = np.clip((t - edge) / max(width, 1e-9), 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def _split_depol_repol(res: CableResult):
    """Split the pseudo-ECG into QRS (depolarisation) and T (repolarisation).

    The two phase masks use smooth ramps rather than hard cut-offs. A hard
    ``where`` boundary leaves a visible step in the synthesised ECG baseline,
    because the pseudo-ECG is not exactly zero at the splice point.
    """
    t = res.t
    qrs_end = np.nanmax(res.activation_time) + 20.0   # ms, end of depolarisation
    repol_start = np.nanmin(res.repol_time) - 15.0    # ms, onset of repolarisation

    pe = res.pseudo_ecg - np.median(res.pseudo_ecg[:5])  # zero the baseline

    qrs = pe * (1.0 - _smoothstep(t, qrs_end, 8.0))
    tw = pe * _smoothstep(t, repol_start, 12.0)
    return qrs, tw


def synthesize_leads(res: CableResult, gain: float | None = None
                     ) -> tuple[dict[str, np.ndarray], float]:
    """Return ({lead_name: signal (mV)}, gain) for all 12 leads.

    If ``gain`` is given it is reused instead of re-normalising. Pass the gain
    from a baseline (no-block) run to subsequent drug runs so that amplitude
    changes (e.g. QRS shrinkage under I_Na block) are preserved rather than
    scaled away.
    """
    qrs, tw = _split_depol_repol(res)

    # Keep both phases in physical pseudo-ECG units so that amplitude changes are
    # preserved: QRS magnitude falls with I_Na block, T timing/shape follows I_Kr.
    # The 1-D pseudo-ECG underweights repolarisation, so the T phase gets a fixed
    # (drug-independent) display boost to bring T/R into a realistic range.
    tw_disp = tw * _T_DISPLAY_BOOST

    X = qrs * _QRS_AXIS[0] + tw_disp * _T_AXIS[0]
    Y = qrs * _QRS_AXIS[1] + tw_disp * _T_AXIS[1]
    Z = qrs * _QRS_AXIS[2] + tw_disp * _T_AXIS[2]
    vcg = np.vstack([X, Y, Z])

    leads: dict[str, np.ndarray] = {}
    for name, c in _DOWER.items():
        leads[name] = np.array(c) @ vcg

    # Goldberger / Einthoven derived leads
    leads["III"] = leads["II"] - leads["I"]
    leads["aVR"] = -(leads["I"] + leads["II"]) / 2.0
    leads["aVL"] = leads["I"] - leads["II"] / 2.0
    leads["aVF"] = leads["II"] - leads["I"] / 2.0

    # amplitude scaling so lead II R-wave is ~physiological. Computed once (from a
    # baseline run) and reused, so drug-induced amplitude changes are not scaled away.
    if gain is None:
        gain = _TARGET_II_R_MV / (np.max(np.abs(leads["II"])) or 1.0)
    for name in leads:
        leads[name] = leads[name] * gain

    return {name: leads[name] for name in LEAD_NAMES}, gain


def qt_interval(res: CableResult) -> float:
    """Approximate QT interval (ms): earliest activation to latest repolarisation."""
    return float(np.nanmax(res.repol_time) - np.nanmin(res.activation_time))
