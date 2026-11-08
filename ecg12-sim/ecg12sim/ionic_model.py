"""Human ventricular myocyte ionic model.

A reduced human-ventricular action-potential model that keeps the two currents
this project focuses on as explicit, separately scalable terms:

    * ``I_Na``  - fast inward sodium current (phase-0 upstroke, m^3 h j gating)
    * ``I_Kr``  - rapid delayed-rectifier potassium current (phase-3 repolarisation)

Gating kinetics for I_Na, I_Kr, I_Ks, I_to, I_CaL and I_K1 follow the
ten Tusscher & Panfilov human ventricular formulations. Intracellular/extracellular
ion concentrations are held fixed and the L-type Ca current uses a simplified
linear driving term, so the only state variables are the membrane voltage and the
gating variables. This keeps the model small and fast while remaining faithful
enough that blocking I_Na or I_Kr produces the expected, well known effects:

    * blocking I_Na  -> slower, smaller phase-0 upstroke (slowed conduction, lower
      QRS amplitude in the ECG)
    * blocking I_Kr  -> prolonged action-potential duration (longer QT interval)

Everything is vectorised over an array of cells so the exact same code drives a
single cell and the 1-D transmural strand used for the pseudo-ECG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants and fixed ionic concentrations
# ---------------------------------------------------------------------------
R = 8314.472   # J / (kmol K)
T = 310.0      # K
F = 96485.3415  # C / mol
RTF = R * T / F  # mV

NAO, NAI = 140.0, 10.0     # mM
KO, KI = 5.4, 140.0        # mM
CAO = 2.0                  # mM
P_KNA = 0.03               # Na permeability of the I_Ks channel

# Reversal potentials (fixed, because concentrations are held constant)
E_NA = RTF * np.log(NAO / NAI)
E_K = RTF * np.log(KO / KI)
E_KS = RTF * np.log((KO + P_KNA * NAO) / (KI + P_KNA * NAI))
E_CA_APP = 65.0  # effective L-type Ca reversal used by the simplified driving term

# Cell-type codes
ENDO, MID, EPI = 0, 1, 2


@dataclass
class Conductances:
    """Maximal conductances (per-pF). Defaults give an epicardial-like AP."""

    GNa: float = 14.838
    GCaL: float = 0.20     # simplified linear L-type scale
    Gto: float = 0.294
    GKr: float = 0.153
    GKs: float = 0.392
    GK1: float = 5.405
    GpK: float = 0.0146
    Gb: float = 0.00029    # small background leak toward E_Na
    PNaK: float = 1.362


@dataclass
class Drug:
    """Multiplicative channel-availability factors (1.0 = no block)."""

    ina_scale: float = 1.0
    ikr_scale: float = 1.0

    @classmethod
    def from_block_percent(cls, ina_block_pct: float = 0.0, ikr_block_pct: float = 0.0) -> "Drug":
        return cls(ina_scale=1.0 - ina_block_pct / 100.0,
                   ikr_scale=1.0 - ikr_block_pct / 100.0)


# Gating-variable layout inside the state matrix (rows = variables, cols = cells)
GATES = ("m", "h", "j", "d", "f", "r", "s", "xr1", "xr2", "xs")
IDX = {name: i for i, name in enumerate(GATES)}
N_STATE = 1 + len(GATES)  # voltage + gates


@dataclass
class CurrentTrace:
    """Container for the individual current components of one cell over time."""

    t: np.ndarray
    V: np.ndarray
    I_Na: np.ndarray
    I_Kr: np.ndarray
    I_Ks: np.ndarray
    I_to: np.ndarray
    I_CaL: np.ndarray
    I_K1: np.ndarray
    extra: dict = field(default_factory=dict)


def initial_state(n_cells: int) -> np.ndarray:
    """Return a resting initial-state matrix of shape (N_STATE, n_cells)."""
    s = np.zeros((N_STATE, n_cells))
    s[0, :] = -86.2          # V
    s[1 + IDX["m"], :] = 0.0
    s[1 + IDX["h"], :] = 0.75
    s[1 + IDX["j"], :] = 0.75
    s[1 + IDX["d"], :] = 0.0
    s[1 + IDX["f"], :] = 1.0
    s[1 + IDX["r"], :] = 0.0
    s[1 + IDX["s"], :] = 1.0
    s[1 + IDX["xr1"], :] = 0.0
    s[1 + IDX["xr2"], :] = 1.0
    s[1 + IDX["xs"], :] = 0.0
    return s


def _gate_inf_tau(V: np.ndarray):
    """Steady-state values and time constants (ms) for every gating variable."""
    inf = {}
    tau = {}

    # --- I_Na activation m ---
    m_inf = 1.0 / (1.0 + np.exp((-56.86 - V) / 9.03)) ** 2
    a_m = 1.0 / (1.0 + np.exp((-60.0 - V) / 5.0))
    b_m = 0.1 / (1.0 + np.exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + np.exp((V - 50.0) / 200.0))
    inf["m"], tau["m"] = m_inf, a_m * b_m

    # --- I_Na inactivation h ---
    h_inf = 1.0 / (1.0 + np.exp((V + 71.55) / 7.43)) ** 2
    a_h = np.where(V >= -40.0, 0.0, 0.057 * np.exp(-(V + 80.0) / 6.8))
    b_h = np.where(V >= -40.0,
                   0.77 / (0.13 * (1.0 + np.exp(-(V + 10.66) / 11.1))),
                   2.7 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.3485 * V))
    inf["h"], tau["h"] = h_inf, 1.0 / (a_h + b_h)

    # --- I_Na slow inactivation j ---
    j_inf = h_inf
    a_j = np.where(V >= -40.0, 0.0,
                   (-2.5428e4 * np.exp(0.2444 * V) - 6.948e-6 * np.exp(-0.04391 * V))
                   * (V + 37.78) / (1.0 + np.exp(0.311 * (V + 79.23))))
    b_j = np.where(V >= -40.0,
                   0.6 * np.exp(0.057 * V) / (1.0 + np.exp(-0.1 * (V + 32.0))),
                   0.02424 * np.exp(-0.01052 * V) / (1.0 + np.exp(-0.1378 * (V + 40.14))))
    inf["j"], tau["j"] = j_inf, 1.0 / (a_j + b_j)

    # --- I_CaL activation d / inactivation f ---
    d_inf = 1.0 / (1.0 + np.exp((-8.0 - V) / 7.5))
    a_d = 1.4 / (1.0 + np.exp((-35.0 - V) / 13.0)) + 0.25
    b_d = 1.4 / (1.0 + np.exp((V + 5.0) / 5.0))
    g_d = 1.0 / (1.0 + np.exp((50.0 - V) / 20.0))
    inf["d"], tau["d"] = d_inf, a_d * b_d + g_d

    f_inf = 1.0 / (1.0 + np.exp((V + 20.0) / 7.0))
    tau_f = (1102.5 * np.exp(-((V + 27.0) ** 2) / 225.0)
             + 200.0 / (1.0 + np.exp((13.0 - V) / 10.0))
             + 180.0 / (1.0 + np.exp((V + 30.0) / 10.0)) + 20.0)
    inf["f"], tau["f"] = f_inf, tau_f

    # --- I_to activation r / inactivation s ---
    inf["r"] = 1.0 / (1.0 + np.exp((20.0 - V) / 6.0))
    tau["r"] = 9.5 * np.exp(-((V + 40.0) ** 2) / 1800.0) + 0.8
    inf["s"] = 1.0 / (1.0 + np.exp((V + 20.0) / 5.0))
    tau["s"] = (85.0 * np.exp(-((V + 45.0) ** 2) / 320.0)
                + 5.0 / (1.0 + np.exp((V - 20.0) / 5.0)) + 3.0)

    # --- I_Kr activation xr1 / inactivation xr2 ---
    inf["xr1"] = 1.0 / (1.0 + np.exp((-26.0 - V) / 7.0))
    a_xr1 = 450.0 / (1.0 + np.exp((-45.0 - V) / 10.0))
    b_xr1 = 6.0 / (1.0 + np.exp((V + 30.0) / 11.5))
    tau["xr1"] = a_xr1 * b_xr1

    inf["xr2"] = 1.0 / (1.0 + np.exp((V + 88.0) / 24.0))
    a_xr2 = 3.0 / (1.0 + np.exp((-60.0 - V) / 20.0))
    b_xr2 = 1.12 / (1.0 + np.exp((V - 60.0) / 20.0))
    tau["xr2"] = a_xr2 * b_xr2

    # --- I_Ks activation xs ---
    inf["xs"] = 1.0 / (1.0 + np.exp((-5.0 - V) / 14.0))
    a_xs = 1400.0 / np.sqrt(1.0 + np.exp((5.0 - V) / 6.0))
    b_xs = 1.0 / (1.0 + np.exp((V - 35.0) / 15.0))
    tau["xs"] = a_xs * b_xs + 80.0

    return inf, tau


def compute_currents(state: np.ndarray, cond: Conductances, drug: Drug,
                     gto_scale: np.ndarray | float = 1.0,
                     gks_scale: np.ndarray | float = 1.0):
    """Return a dict of ionic currents (per-pF) for the current state.

    ``gto_scale`` / ``gks_scale`` let the caller impose transmural heterogeneity
    (endo/mid/epi) without changing the base conductances.
    """
    V = state[0]
    g = {name: state[1 + IDX[name]] for name in GATES}

    I_Na = cond.GNa * drug.ina_scale * g["m"] ** 3 * g["h"] * g["j"] * (V - E_NA)
    I_CaL = cond.GCaL * g["d"] * g["f"] * (V - E_CA_APP)
    I_to = cond.Gto * gto_scale * g["r"] * g["s"] * (V - E_K)
    I_Kr = cond.GKr * drug.ikr_scale * np.sqrt(KO / 5.4) * g["xr1"] * g["xr2"] * (V - E_K)
    I_Ks = cond.GKs * gks_scale * g["xs"] ** 2 * (V - E_KS)

    a_k1 = 0.1 / (1.0 + np.exp(0.06 * (V - E_K - 200.0)))
    b_k1 = ((3.0 * np.exp(0.0002 * (V - E_K + 100.0)) + np.exp(0.1 * (V - E_K - 10.0)))
            / (1.0 + np.exp(-0.5 * (V - E_K))))
    xk1 = a_k1 / (a_k1 + b_k1)
    I_K1 = cond.GK1 * np.sqrt(KO / 5.4) * xk1 * (V - E_K)

    I_pK = cond.GpK * (V - E_K) / (1.0 + np.exp((25.0 - V) / 5.98))
    I_NaK = (cond.PNaK * KO / (KO + 1.0) * NAI / (NAI + 40.0)
             / (1.0 + 0.1245 * np.exp(-0.1 * V / RTF) + 0.0353 * np.exp(-V / RTF)))
    I_bNa = cond.Gb * (V - E_NA)

    return {
        "I_Na": I_Na, "I_CaL": I_CaL, "I_to": I_to, "I_Kr": I_Kr,
        "I_Ks": I_Ks, "I_K1": I_K1, "I_pK": I_pK, "I_NaK": I_NaK, "I_bNa": I_bNa,
    }


def step(state: np.ndarray, dt: float, I_stim: np.ndarray | float,
         cond: Conductances, drug: Drug,
         gto_scale: np.ndarray | float = 1.0,
         gks_scale: np.ndarray | float = 1.0) -> tuple[np.ndarray, dict]:
    """Advance the state one time step (Rush-Larsen for gates, forward Euler for V)."""
    V = state[0]
    cur = compute_currents(state, cond, drug, gto_scale, gks_scale)
    I_ion = sum(cur.values())

    new = state.copy()
    new[0] = V + dt * (-(I_ion) + I_stim)

    inf, tau = _gate_inf_tau(V)
    for name in GATES:
        gi = state[1 + IDX[name]]
        new[1 + IDX[name]] = inf[name] + (gi - inf[name]) * np.exp(-dt / tau[name])

    return new, cur
