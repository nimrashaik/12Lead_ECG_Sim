"""ecg12sim - a compact human-ventricular 12-lead ECG simulator.

Generates a 12-lead ECG from a transmural ionic-model strand and shows how
scaling the fast sodium current (I_Na) and the rapid delayed-rectifier
potassium current (I_Kr) changes the action potential and the ECG.
"""

from .ionic_model import Conductances, Drug
from .cable import CableConfig, simulate_cable
from .leads import LEAD_NAMES, synthesize_leads, qt_interval
from .simulate import run_single_cell, run_12lead, CellTrace, EcgResult
from .tracking import (
    analyse, conduction_velocity, TrackingReport, DepthBand, band_profile,
    cable_band_profile, format_band_profile,
)
from .timeline import (
    TimelinePoint, CurrentSnapshot, Landmark,
    build_timeline, timeline_from_cable, timeline_from_reduced, find_landmarks,
    format_timeline, format_landmarks,
)
from .mesh import (
    VentricularMesh, load_ventricular_mesh, fetch_mesh, region_from_rho,
    CITATION as MESH_CITATION,
)
from .eikonal import (
    Velocities, ActivationResult, solve_activation, calibrate_metrication,
    measure_metrication, format_activation_report,
)
from .templates import (
    TemplateBank, Heterogeneity, build_template_bank, calibrate_dispersion,
)
from .mesh_sim import (
    MeshBeat, ReducedCurrent, run_mesh_beat, nodal_volumes, mesh_band_profile,
)
from .sweep import (
    APMetrics, SweepResult, SWEEP_SCALES, run_ap_sweep, run_complete_ap,
    format_sweep_table, format_complete_ap,
)
from .forward import (
    AnatomicalFrame, Conductivity, LeadField, BemResult, build_frame,
    build_lead_field, lead_field_weights, electrode_positions,
    electrode_potentials, derive_12_leads, run_12lead_mesh, run_12lead_bem,
    surface_potentials_unbounded, check_morphology, format_morphology,
)
from .torso import (
    Surface, make_torso, make_sphere, bem_matrix, solve_surface,
    validate_sphere, format_sphere_check,
)

__all__ = [
    "Conductances", "Drug", "CableConfig", "simulate_cable",
    "LEAD_NAMES", "synthesize_leads", "qt_interval",
    "run_single_cell", "run_12lead", "CellTrace", "EcgResult",
    "analyse", "conduction_velocity", "TrackingReport",
    "TimelinePoint", "CurrentSnapshot", "Landmark",
    "build_timeline", "timeline_from_cable", "timeline_from_reduced",
    "find_landmarks", "format_timeline", "format_landmarks",
    "VentricularMesh", "load_ventricular_mesh", "fetch_mesh", "region_from_rho",
    "MESH_CITATION",
    "Velocities", "ActivationResult", "solve_activation",
    "calibrate_metrication", "measure_metrication", "format_activation_report",
    "TemplateBank", "Heterogeneity", "build_template_bank",
    "calibrate_dispersion",
    "MeshBeat", "ReducedCurrent", "run_mesh_beat", "nodal_volumes",
    "mesh_band_profile",
    "APMetrics", "SweepResult", "SWEEP_SCALES", "run_ap_sweep",
    "run_complete_ap", "format_sweep_table", "format_complete_ap",
    "DepthBand", "band_profile", "cable_band_profile", "format_band_profile",
    "AnatomicalFrame", "Conductivity", "LeadField", "BemResult",
    "build_frame", "build_lead_field", "lead_field_weights",
    "electrode_positions", "electrode_potentials", "derive_12_leads",
    "run_12lead_mesh", "run_12lead_bem", "surface_potentials_unbounded",
    "check_morphology", "format_morphology",
    "Surface", "make_torso", "make_sphere", "bem_matrix", "solve_surface",
    "validate_sphere", "format_sphere_check",
]

__version__ = "0.1.0"
