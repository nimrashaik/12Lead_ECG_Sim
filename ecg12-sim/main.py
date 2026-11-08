#!/usr/bin/env python3
"""Command-line entry point: generate a 12-lead ECG and track I_Na / I_Kr.

By default nothing is blocked or altered. The simulation runs one normal beat and
*tracks* where and when the fast sodium current (I_Na) and the rapid
delayed-rectifier potassium current (I_Kr) flow across the ventricular wall,
relating both to the ECG the same beat produces.

Examples
--------
Normal beat: 12-lead ECG plus the I_Na / I_Kr tracking views::

    python main.py

Longer beat, different output folder::

    python main.py --duration 600 --outdir out

Optionally also apply a channel block and compare against the normal beat
(off by default)::

    python main.py --ikr-block 70
"""

from __future__ import annotations

import argparse
import os

from ecg12sim import ionic_model as im
from ecg12sim import leads as ld
from ecg12sim import plotting, timeline, tracking
from ecg12sim.cable import CableConfig
from ecg12sim.simulate import run_12lead, run_single_cell


def eikonal_defaults():
    """Library defaults, so the CLI cannot drift away from them."""
    from ecg12sim.eikonal import Velocities
    return Velocities()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a 12-lead ECG and track I_Na / I_Kr across the wall.")
    p.add_argument("--duration", type=float, default=500.0,
                   help="Simulated time in ms. Default 500.")
    p.add_argument("--cells", type=int, default=100,
                   help="Number of cells across the wall (cable model). "
                        "Default 100.")
    p.add_argument("--model", choices=("cable", "mesh"), default="cable",
                   help="cable: 1-D transmural strand, emergent conduction and "
                        "repolarisation, runs in seconds. mesh: anatomically "
                        "accurate biventricular mesh with reaction-eikonal "
                        "propagation. Default cable.")

    mg = p.add_argument_group("mesh model (--model mesh)")
    mg.add_argument("--mesh-case", default="01",
                    help="Cohort case 01-20 to fetch/use. Default 01.")
    mg.add_argument("--mesh-file", default=None,
                    help="Path to a cohort .vtk, bypassing --mesh-case.")
    mg.add_argument("--mesh-dir", default="data/meshes",
                    help="Where meshes are downloaded/cached. "
                         "Default data/meshes.")
    mg.add_argument("--templates", type=int, default=15,
                    help="Transmural depths in the template bank. Default 15.")
    mg.add_argument("--cv-fibre", type=float, default=65.0,
                    help="Conduction velocity along fibre, cm/s. Default 65.")
    mg.add_argument("--cv-transverse", type=float, default=25.0,
                    help="Transverse conduction velocity, cm/s. Default 25.")
    mg.add_argument("--endo-cv", type=float,
                    default=eikonal_defaults().endo_layer_cv,
                    help="Fast endocardial (Purkinje surrogate) shell velocity, "
                         "cm/s; 0 disables. Lower values leave activation "
                         "apex-to-base dominated and invert the frontal-plane "
                         "leads.")
    mg.add_argument("--unbounded", action="store_true",
                    help="Use the unbounded-medium forward model instead of "
                         "the bounded torso. Limb leads come out ~25x too "
                         "small; kept for comparison only.")
    mg.add_argument("--torso-theta", type=int, default=24,
                    help="Circumferential resolution of the torso surface. "
                         "Default 24.")
    mg.add_argument("--calibrate-metrication", action="store_true",
                    help="Measure and correct the graph's path-length bias "
                         "so realised conduction velocity matches --cv-fibre.")
    p.add_argument("--outdir", default="examples",
                   help="Directory for the PNG figures. Default ./examples.")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write PNG files (compute and print metrics only).")

    sw = p.add_argument_group("single-cell action potential and conductance sweeps")
    sw.add_argument("--ap-sweep", action="store_true",
                    help="Show the complete single-cell action potential and "
                         "how it changes when I_Na and I_Kr are scaled "
                         "separately across levels above and below baseline.")
    sw.add_argument("--sweep-cell", default="epi",
                    choices=("endo", "mid", "epi"),
                    help="Cell type for the sweeps. Default epi.")
    sw.add_argument("--sweep-levels", type=int, default=10,
                    help="Number of conductance levels, geometric about "
                         "baseline. Default 10.")
    sw.add_argument("--sweep-range", type=float, default=4.0,
                    help="Largest scale factor; the smallest is its "
                         "reciprocal-ish counterpart. Default 4.0.")

    p.add_argument("--variation", action="store_true",
                   help="Print the transmural variation of I_Na and I_Kr as a "
                        "numeric table: peak and peak time by wall depth, with "
                        "the span across the wall.")
    p.add_argument("--variation-bands", type=int, default=9,
                   help="Depth bands in the variation table. Default 9.")

    tl = p.add_argument_group("instant-by-instant I_Na / I_Kr readout")
    tl.add_argument("--no-timeline", action="store_true",
                    help="Suppress the full per-instant table (keep the landmarks).")
    tl.add_argument("--timeline-step", type=float, default=None,
                    help="Uniform readout stride in ms. Default: adaptive "
                         "(1 ms through the QRS, 10 ms elsewhere).")
    tl.add_argument("--timeline-lead", default="II", choices=ld.LEAD_NAMES,
                    help="Which lead to show in the readout's ECG column. Default II.")

    g = p.add_argument_group("optional channel block (off by default)")
    g.add_argument("--ina-block", type=float, default=0.0,
                   help="I_Na block percent 0-100. Default 0 (no block).")
    g.add_argument("--ikr-block", type=float, default=0.0,
                   help="I_Kr block percent 0-100. Default 0 (no block).")
    return p.parse_args(argv)


def run_mesh_model(args):
    """Run the beat on an anatomically accurate biventricular mesh."""
    from ecg12sim import eikonal, mesh as meshmod, mesh_sim, templates

    path = args.mesh_file
    if path is None:
        print(f"Locating cohort mesh {args.mesh_case} "
              f"(downloads ~90 MB on first use)...")
        path = meshmod.fetch_mesh(args.mesh_case, dest_dir=args.mesh_dir)

    print(f"Loading {path} ...")
    mesh = meshmod.load_ventricular_mesh(path)
    print(mesh.summary())
    print()
    print("Mesh data: " + meshmod.CITATION)

    correction = 1.0
    if args.calibrate_metrication:
        print()
        print("Calibrating graph path-length bias ...")
        correction = eikonal.calibrate_metrication(mesh, speed=args.cv_fibre)
        print(f"  correction factor {correction:.4f}")

    vel = eikonal.Velocities(
        along_fibre=args.cv_fibre,
        across_sheet=args.cv_transverse,
        across_normal=args.cv_transverse,
        endo_layer_cv=args.endo_cv,
        metrication_correction=correction,
    )

    print()
    print("Solving activation ...")
    act = eikonal.solve_activation(mesh, vel)
    stats = eikonal.measure_metrication(mesh, speed=args.cv_fibre)
    print()
    print(eikonal.format_activation_report(mesh, act, stats))

    print()
    print(f"Building template bank ({args.templates} transmural depths) ...")
    drug = im.Drug.from_block_percent(args.ina_block, args.ikr_block)
    bank = templates.build_template_bank(drug=drug, n_templates=args.templates,
                                         duration=max(600.0, args.duration))

    print("Assembling beat ...")
    beat = mesh_sim.run_mesh_beat(mesh, drug=drug, bank=bank, activation=act)
    print()
    print(beat.summary())

    if args.variation:
        print()
        print(tracking.format_band_profile(
            mesh_sim.mesh_band_profile(beat, n_bands=args.variation_bands),
            title="Transmural variation of I_Na and I_Kr (anatomical mesh)"))

    from ecg12sim import forward
    print()
    if args.unbounded:
        print("Building forward model, UNBOUNDED medium ...")
        mframe, lf, phi, leads = forward.run_12lead_mesh(beat)
        print()
        print(mframe.summary())
        print()
        print("Warning: with no torso boundary a dipole field falls as 1/r^2,")
        print("so the limb electrodes at 28-40 cm read roughly 4% of the")
        print("precordial amplitude, where a real ECG has them comparable.")
        print("Only the precordial leads are quantitatively meaningful here.")
    else:
        print("Building bounded-torso forward model (boundary element) ...")
        bem = forward.run_12lead_bem(beat, n_theta=args.torso_theta)
        leads = bem.leads
        print()
        print(bem.frame.summary())
        print()
        print(bem.summary())

    print()
    print("Lead amplitudes, mV (genuine potential differences, no fitted axes):")
    for n in forward.LEAD_NAMES:
        s = leads[n]
        print(f"  {n:<3s} min {s.min():+8.4f}  max {s.max():+8.4f}  "
              f"p2p {s.max() - s.min():7.4f}")

    import numpy as _np
    limb = _np.mean([_np.ptp(leads[n]) for n in ("I", "II", "III")])
    prec = _np.mean([_np.ptp(leads[n]) for n in ("V2", "V3", "V4")])
    print(f"  limb/precordial peak-to-peak ratio: {limb / prec:.3f} "
          f"(a real ECG is roughly 0.3-1.0)")

    print()
    print(forward.format_morphology(forward.check_morphology(
        leads, beat.qrs_window, beat.t_window, beat.t)))

    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        fig_path = os.path.join(args.outdir, "ecg12_mesh.png")
        kind = "unbounded medium" if args.unbounded else "bounded torso"
        plotting.plot_12lead_mesh(
            beat, leads,
            f"12-lead ECG - anatomical mesh, {kind} "
            f"({mesh.n_nodes:,} nodes)", path=fig_path)
        print()
        print("Saved figure:")
        print("  " + fig_path)

    lead_name = args.timeline_lead
    label = f"lead {lead_name}"
    rows = timeline.timeline_from_reduced(
        beat.t, leads[lead_name], beat.ina, beat.ikr,
        beat.qrs_window, beat.t_window, step=args.timeline_step)
    marks = timeline.find_landmarks(rows)
    print()
    print(timeline.format_landmarks(marks, signal_label=label))
    if not args.no_timeline:
        print()
        print(timeline.format_timeline(rows, signal_label=label))

    return beat, leads


def run_ap_sweep(args):
    """Complete single-cell action potential, then scale I_Na and I_Kr."""
    import numpy as np
    from ecg12sim import sweep as sw

    # Geometric ladder spanning 1/range to range, with baseline included
    # exactly. Geometric rather than linear so a halving and a doubling sit
    # equally far from baseline.
    n = max(int(args.sweep_levels), 3)
    hi = np.log2(max(args.sweep_range, 1.01))
    half = n // 2
    exps = np.concatenate([
        np.linspace(-hi, 0.0, half + 1)[:-1],     # below baseline
        [0.0],                                    # baseline exactly
        np.linspace(0.0, hi, n - half)[1:],       # above baseline
    ])
    scales = tuple(float(2.0 ** e) for e in exps)

    print(f"Simulating one {args.sweep_cell} ventricular myocyte "
          f"(baseline, no scaling)...")
    t, V, traces, met = sw.run_complete_ap(cell_type=args.sweep_cell,
                                           duration=args.duration)
    print()
    print(sw.format_complete_ap(t, V, traces, met, args.sweep_cell))

    results = {}
    for which in ("I_Na", "I_Kr"):
        print()
        print(f"Scaling {which} across {len(scales)} levels "
              f"({min(scales):.2f}x to {max(scales):.2f}x)...")
        res = sw.run_ap_sweep(which, scales=scales,
                              cell_type=args.sweep_cell,
                              duration=args.duration)
        results[which] = res
        print()
        print(sw.format_sweep_table(res))

    outputs = []
    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        p1 = os.path.join(args.outdir, "ap_complete.png")
        plotting.plot_complete_ap(t, V, traces, met, args.sweep_cell, path=p1)
        p2 = os.path.join(args.outdir, "ap_sweep_ina_ikr.png")
        plotting.plot_ap_sweeps(results["I_Na"], results["I_Kr"], path=p2)
        outputs = [p1, p2]
        print()
        print("Saved figures:")
        for o in outputs:
            print("  " + o)
    return results


def main(argv=None):
    args = parse_args(argv)
    for v, label in ((args.ina_block, "--ina-block"), (args.ikr_block, "--ikr-block")):
        if not 0.0 <= v <= 100.0:
            raise SystemExit(f"{label} must be between 0 and 100 (got {v}).")

    if args.ap_sweep:
        run_ap_sweep(args)
        return

    if args.model == "mesh":
        run_mesh_model(args)
        return

    has_drug = args.ina_block > 0 or args.ikr_block > 0
    cfg = CableConfig(duration=args.duration, n_cells=args.cells)

    # ---- normal beat: the thing we track ----
    print("Simulating one normal beat (no channel block)...")
    normal = im.Drug()
    ecg = run_12lead(normal, cfg=cfg)
    rep = tracking.analyse(ecg.cable)

    print()
    print(tracking.format_report(rep, ecg.qt))

    if args.variation:
        print()
        print(tracking.format_band_profile(
            tracking.cable_band_profile(ecg.cable, n_bands=args.variation_bands),
            title="Transmural variation of I_Na and I_Kr (1-D strand)"))

    # ---- instant-by-instant readout of both currents as the ECG forms ----
    lead_name = args.timeline_lead
    rows = timeline.timeline_from_cable(
        ecg.cable, ecg.leads[lead_name],
        rep.qrs_window, rep.t_window, step=args.timeline_step)
    marks = timeline.find_landmarks(rows)

    print()
    print(timeline.format_landmarks(marks, signal_label=f"lead {lead_name}"))
    if not args.no_timeline:
        print()
        print(timeline.format_timeline(rows, signal_label=f"lead {lead_name}"))

    outputs = []
    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        jobs = [
            ("ecg12_normal.png",
             lambda p: plotting.plot_12lead(ecg, "12-lead ECG - normal beat",
                                            path=p, rep=rep)),
            ("track_current_maps.png",
             lambda p: plotting.plot_current_maps(ecg, rep, path=p)),
            ("track_transmural.png",
             lambda p: plotting.plot_transmural_traces(ecg, rep, path=p)),
            ("track_ecg_correlation.png",
             lambda p: plotting.plot_current_ecg_correlation(ecg, rep, path=p)),
            ("ecg_ina_ikr_combined.png",
             lambda p: plotting.plot_combined_waveforms(ecg, rep, path=p)),
        ]
        for name, fn in jobs:
            path = os.path.join(args.outdir, name)
            fn(path)
            outputs.append(path)

    # ---- optional: compare a blocked beat against the normal one ----
    if has_drug:
        print()
        print(f"Also simulating a blocked beat for comparison "
              f"(I_Na {args.ina_block:.0f}%, I_Kr {args.ikr_block:.0f}%)...")
        drug = im.Drug.from_block_percent(args.ina_block, args.ikr_block)
        drug_ecg = run_12lead(drug, cfg=cfg, gain=ecg.gain)
        drug_rep = tracking.analyse(drug_ecg.cable)
        print(f"  QT        : {ecg.qt:.0f} ms -> {drug_ecg.qt:.0f} ms "
              f"({drug_ecg.qt - ecg.qt:+.0f} ms)")
        print(f"  I_Na peak : {rep.ina_peak_global:.1f} -> "
              f"{drug_rep.ina_peak_global:.1f} pA/pF")
        print(f"  I_Kr peak : {rep.ikr_peak_global:.3f} -> "
              f"{drug_rep.ikr_peak_global:.3f} pA/pF")
        print(f"  CV        : {rep.conduction_velocity:.1f} -> "
              f"{drug_rep.conduction_velocity:.1f} cm/s")
        if not args.no_save:
            tag = f"ina{int(args.ina_block)}_ikr{int(args.ikr_block)}"
            
            # p = os.path.join(args.outdir, f"compare_{tag}.png")
            # plotting.plot_ecg_comparison(ecg, drug_ecg, path=p)
            # outputs.append(p)
            
            p12 = os.path.join(
            args.outdir, f"ecg12_comparison_ina{int(args.ina_block)}_ikr{int(args.ikr_block)}.png"
            )
            plotting.plot_12lead_overlay(
                ecg,
                drug_ecg,
                title="12-lead ECG: normal vs channel block",
                ina_block=args.ina_block,
                ikr_block=args.ikr_block,
                path=p12
                )
            
            outputs.append(p12)
            
            
            from ecg12sim import sweep as sw
            ap_base = sw.run_complete_ap(
            cell_type=args.sweep_cell,
            duration=args.duration
                )

            ap_drug = sw.run_complete_ap(
            cell_type=args.sweep_cell,
            duration=args.duration,
            drug=drug
            )

            t_base, v_base, tr_base, met_base = ap_base
            t_drug, v_drug, tr_drug, met_drug = ap_drug
            
            # pbase = os.path.join(args.outdir, "ap_complete_normal.png")
            # plotting.plot_complete_ap(
            # t_base, v_base, tr_base, met_base,
            # args.sweep_cell,
            # path=pbase
            # )
            # outputs.append(pbase)

            # Save drug AP
            # tag = f"ina{int(args.ina_block)}_ikr{int(args.ikr_block)}"
            # pdrug = os.path.join(args.outdir, f"ap_complete_{tag}.png")

            # plotting.plot_complete_ap(
            # t_drug, v_drug, tr_drug, met_drug,
            # args.sweep_cell,
            # path=pdrug
            # )
            # outputs.append(pdrug)
            # Save overlay comparison
            pac = os.path.join(
            args.outdir,
            f"ap_complete_comparison_{tag}.png"
            )

            plotting.plot_complete_ap_comparison(
            t_base, v_base, tr_base, met_base,
            t_drug, v_drug, tr_drug, met_drug,
            args.sweep_cell,
            label=(
            f"I_Na block {args.ina_block:.0f}%, "
            f"I_Kr block {args.ikr_block:.0f}%"
            ),
            path=pac
            )
            outputs.append(pac)
           
            



    if outputs:
        print("\nSaved figures:")
        for o in outputs:
            print("  " + o)


if __name__ == "__main__":
    main()
