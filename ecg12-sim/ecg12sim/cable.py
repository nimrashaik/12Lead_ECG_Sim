"""1-D transmural strand (monodomain cable) and pseudo-ECG computation.

A single line of ventricular cells is arranged across the wall from endocardium
to epicardium with a mid-myocardial (M-cell) region in between. The endocardial
end is stimulated, a wavefront propagates across the wall through diffusive
coupling, and the resulting transmembrane-voltage field ``Vm(x, t)`` is turned
into an extracellular *pseudo-ECG* using the standard spatial-gradient integral
(Gima & Rudy, 2002).

The transmural APD gradient (M cells repolarise last) produces an upright T wave,
and the model's I_Na / I_Kr scaling flows straight through:

    * I_Na block -> slower wavefront, smaller/wider depolarisation complex
    * I_Kr block -> delayed repolarisation -> later T wave, longer QT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ionic_model as im


@dataclass
class CableConfig:
    n_cells: int = 100
    dx: float = 0.015          # cm, inter-cell spacing
    D: float = 0.0013          # cm^2/ms, diffusion coefficient
    dt: float = 0.02           # ms
    duration: float = 500.0    # ms
    stim_amp: float = 52.0     # pA/pF
    stim_dur: float = 1.0      # ms
    stim_cells: int = 3        # endocardial cells stimulated
    electrode_gap: float = 2.0  # cm, electrode distance beyond epicardium


@dataclass
class CableResult:
    t: np.ndarray              # (nt,)
    x: np.ndarray              # (n_cells,) position in cm
    Vm: np.ndarray             # (nt, n_cells)
    pseudo_ecg: np.ndarray     # (nt,)
    activation_time: np.ndarray  # (n_cells,) ms
    repol_time: np.ndarray       # (n_cells,) ms
    region: np.ndarray           # (n_cells,) endo/mid/epi codes
    # Spatiotemporal current fields, recorded for every cell at every sample.
    # These are what the "tracking" views visualise: I_Na marks the travelling
    # depolarisation wavefront, I_Kr the spatially dispersed repolarisation.
    I_Na: np.ndarray = None    # (nt, n_cells) pA/pF
    I_Kr: np.ndarray = None    # (nt, n_cells) pA/pF


def _region_scales(n_cells: int):
    """Assign endo/mid/epi regions and their I_Ks / I_to heterogeneity scales."""
    region = np.empty(n_cells, dtype=int)
    gks = np.ones(n_cells)
    gto = np.ones(n_cells)
    for i in range(n_cells):
        frac = i / (n_cells - 1)
        if frac < 0.30:                 # endocardium
            region[i] = im.ENDO
            gks[i] = 1.0
            gto[i] = 0.25               # small I_to -> no epi-like notch
        elif frac < 0.60:               # mid-myocardium (M cells)
            region[i] = im.MID
            gks[i] = 0.25               # weak I_Ks -> long APD, I_Kr sensitive
            gto[i] = 1.0
        else:                           # epicardium
            region[i] = im.EPI
            gks[i] = 1.0
            gto[i] = 1.0
    return region, gks, gto


def simulate_cable(cfg: CableConfig, drug: im.Drug,
                   cond: im.Conductances | None = None,
                   record_every: int = 5) -> CableResult:
    """Integrate the strand and return voltages plus the pseudo-ECG."""
    cond = cond or im.Conductances()
    n = cfg.n_cells
    region, gks, gto = _region_scales(n)

    state = im.initial_state(n)
    coeff = cfg.D / cfg.dx ** 2  # 1/ms

    nt_full = int(round(cfg.duration / cfg.dt))
    rec_idx = list(range(0, nt_full, record_every))
    nt = len(rec_idx)

    x = np.arange(n) * cfg.dx
    Vm_rec = np.zeros((nt, n))
    ina_rec = np.zeros((nt, n))
    ikr_rec = np.zeros((nt, n))
    t_rec = np.zeros(nt)

    activation = np.full(n, np.nan)
    repol = np.full(n, np.nan)
    prevV = state[0].copy()

    # electrode sits beyond the epicardial end, on the strand axis
    x_elec = x[-1] + cfg.electrode_gap
    inv_dist = 1.0 / np.abs(x - x_elec)      # geometric weight ~ 1/r
    pseudo = np.zeros(nt)

    r = 0
    for k in range(nt_full):
        t = k * cfg.dt
        V = state[0]

        # --- diffusion (no-flux boundaries) ---
        lap = np.empty(n)
        lap[1:-1] = V[:-2] - 2.0 * V[1:-1] + V[2:]
        lap[0] = V[1] - V[0]
        lap[-1] = V[-2] - V[-1]
        diff = coeff * lap

        # --- stimulus on endocardial cells ---
        stim = np.zeros(n)
        if t < cfg.stim_dur:
            stim[:cfg.stim_cells] = cfg.stim_amp

        # --- ionic currents + membrane update ---
        cur = im.compute_currents(state, cond, drug, gto_scale=gto, gks_scale=gks)
        I_ion = sum(cur.values())
        newV = V + cfg.dt * (-(I_ion) + diff + stim)

        # --- gates (Rush-Larsen) ---
        inf, tau = im._gate_inf_tau(V)
        new = state.copy()
        new[0] = newV
        for name in im.GATES:
            gi = state[1 + im.IDX[name]]
            new[1 + im.IDX[name]] = inf[name] + (gi - inf[name]) * np.exp(-cfg.dt / tau[name])

        # --- activation / repolarisation bookkeeping (threshold -40 mV) ---
        crossed_up = np.isnan(activation) & (prevV < -40.0) & (newV >= -40.0)
        activation[crossed_up] = t
        crossed_dn = (~np.isnan(activation) & np.isnan(repol)
                      & (prevV >= -40.0) & (newV < -40.0))
        repol[crossed_dn] = t
        prevV = newV

        state = new

        if r < nt and k == rec_idx[r]:
            Vm_rec[r] = V
            ina_rec[r] = cur["I_Na"]
            ikr_rec[r] = cur["I_Kr"]
            t_rec[r] = t
            # pseudo-ECG: -integral( dVm/dx * d(1/r)/dx ) dx, discretised
            gradV = np.gradient(V, cfg.dx)
            grad_inv = np.gradient(inv_dist, cfg.dx)
            pseudo[r] = -np.sum(gradV * grad_inv) * cfg.dx
            r += 1

    return CableResult(t=t_rec, x=x, Vm=Vm_rec, pseudo_ecg=pseudo,
                       activation_time=activation, repol_time=repol, region=region,
                       I_Na=ina_rec, I_Kr=ikr_rec)
