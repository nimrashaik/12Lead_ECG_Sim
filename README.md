# 12-Lead ECG Simulator

A 12-lead ECG simulator built from ionic-model myocardium on an image-derived
four-chamber heart mesh. It produces P waves, QRS complexes and T waves, runs
multi-beat rhythm strips at a prescribed heart rate, and tracks where and when
the fast sodium current (I_Na) and the rapid delayed-rectifier potassium current
(I_Kr) flow through the wall.

Dependencies are numpy, scipy and matplotlib. Nothing else.

```bash
python main.py --model mesh --rhythm        # 10-second 12-lead ECG at 72 bpm
```

![10-second 12-lead ECG at 72 bpm](examples/rhythm_72bpm/ecg12_rhythm_72bpm.png)

Twelve beats, one every 833 ms, in the layout a clinical machine prints: each
lead over its own 2.5-second quarter, with lead II running the full ten seconds
along the bottom. The P waves are the small deflections before each QRS, clearest
in II, III, aVF and V1-V3.

Those P waves are atrial in origin, which the detail figure shows by drawing each
source separately against their sum. In the shaded bands the recorded trace lies
exactly on the atrial curve while the ventricular curve is flat:

![P wave detail](examples/rhythm_72bpm/ecg_p_wave_detail.png)

And the activation sequence that produces them: the sinoatrial node sits on the
subject's right and superior, and depolarisation sweeps left and inferior over
95 ms, which is what gives the P wave its +31° axis and its 96 ms duration.

![Atrial activation](examples/rhythm_72bpm/atrial_activation.png)

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [The three models](#the-three-models)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [Using it as a library](#using-it-as-a-library)
- [How it works](#how-it-works)
- [What is validated, and how](#what-is-validated-and-how)
- [Known limitations](#known-limitations)
- [Module map](#module-map)
- [Data source and citation](#data-source-and-citation)

---

## Install

Developed and tested on Python 3.13. **numpy 2.0 or newer is required** (the
code uses `np.trapezoid`, added in numpy 2.0).

```bash
cd project_sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Verify:

```bash
.venv/bin/python -c "import ecg12sim; print(ecg12sim.__version__)"
```

Every command below is written as `python main.py ...`. If you used the venv
without activating it, write `.venv/bin/python main.py ...` instead.

### The anatomical mesh is downloaded on first use

The 1-D cable model needs no data. The mesh models need one case from the Rodero
four-chamber cohort, which `main.py` fetches automatically into `data/meshes/`
(gitignored) and verifies against the record's published MD5:

| | size | note |
|---|---|---|
| `01.tar.gz` | 91 MB | the download; safe to delete after extraction |
| `01.vtk` | 260 MB | extracted ASCII VTK |
| `01.ventricles.npz` | 69 MB | cache, built on first use |
| `01.atria.npz` | 5 MB | cache, built on first use |

The first mesh run therefore spends a few minutes downloading. After the caches
exist you can delete the `.tar.gz`, but **keep the `.vtk`**: the caches are
rebuilt from it if you change cache versions or pass `rebuild=True`.

To pre-fetch a case without running a simulation:

```bash
python -c "from ecg12sim import fetch_mesh; print(fetch_mesh('01'))"
```

Cases `01` through `20` are available via `--mesh-case`.

---

## Quick start

**A 10-second 12-lead ECG at 72 bpm, with P waves.** This is the headline
output: twelve beats, one every 833 ms, atria and ventricles both contributing.

```bash
python main.py --model mesh --rhythm --outdir out
```

**One cardiac cycle** (atria and ventricles, no `--rhythm`):

```bash
python main.py --model mesh --outdir out
```

**A different rate and PR interval:**

```bash
python main.py --model mesh --rhythm --heart-rate 50 --pr-interval 200 --outdir out
```

**Fast, no download** — the 1-D transmural strand, which runs in about 4 seconds
and is the right tool for ion-current questions:

```bash
python main.py --variation
```

**Drug effect on the action potential:**

```bash
python main.py --ap-sweep                  # scale I_Na and I_Kr, see APD change
python main.py --ikr-block 70              # 70% I_Kr block, QT prolongation
```

**Print metrics without writing figures:** add `--no-save`. **Suppress the long
per-instant current table:** add `--no-timeline`.

> **Pass `--outdir` on anything that saves.** The default is `examples/`, which
> already holds committed reference figures, and a run will overwrite any of
> them whose names it reuses. The examples above write to `out/` for that
> reason.

### Runtimes measured on an Apple-silicon laptop

| command | time | peak memory |
|---|---|---|
| `--model cable` | 4 s | small |
| `--model mesh --no-atria` | 67 s | 2.6 GB |
| `--model mesh` (one cycle, with atria) | 84 s | 2.6 GB |
| `--model mesh --rhythm` (10 s strip) | 84 s | 2.6 GB |

A 10-second strip costs the same as one cycle. That is the point of the
superposition described under [How it works](#how-it-works): the expensive part
is the forward solve for two single beats, and the strip is assembled from
shifted copies.

**2.6 GB of RAM is a hard requirement for the mesh models.** The lead field
touches 1.85 million ventricular and 397 thousand atrial tetrahedra.

---

## The three models

| | `--model cable` | `--model mesh --no-atria` | `--model mesh` |
|---|---|---|---|
| geometry | 1-D strand, 1.5 cm | 355,095-node biventricular mesh | plus 97,985 atrial nodes |
| P wave | no | no | **yes** |
| leads | one axis, projected to 12 | 12 independent | 12 independent |
| conduction | emergent, diffusive | anisotropic eikonal | plus isotropic atrial |
| repolarisation | emergent | prescribed by template bank | same |
| multi-beat | no | no | **yes, `--rhythm`** |
| data needed | none | cohort mesh | cohort mesh |

The cable model is not a lesser version of the mesh model; it is the only one
where repolarisation *emerges* from coupling between cells rather than being
written by a template bank, so it is the better model for questions about APD
and the T wave. The mesh model is the only one that produces genuinely
independent leads, P waves and a rhythm.

---

## Command-line reference

### Model selection

| flag | default | meaning |
|---|---|---|
| `--model {cable,mesh}` | `cable` | which model to run |
| `--duration MS` | 500 | simulated time for the cable model and template length |
| `--cells N` | 100 | cells across the wall, cable model only |
| `--outdir DIR` | `examples` | where PNGs are written |
| `--no-save` | off | compute and print only |

### Mesh model (`--model mesh`)

| flag | default | meaning |
|---|---|---|
| `--mesh-case NN` | `01` | cohort case 01-20, fetched if absent |
| `--mesh-file PATH` | — | use a specific `.vtk`, bypassing `--mesh-case` |
| `--mesh-dir DIR` | `data/meshes` | download and cache location |
| `--templates N` | 15 | transmural depths in the ventricular template bank |
| `--cv-fibre CM_S` | 65 | ventricular conduction velocity along fibre |
| `--cv-transverse CM_S` | 25 | ventricular transverse conduction velocity |
| `--endo-cv CM_S` | 600 | fast endocardial shell, a Purkinje surrogate; 0 disables |
| `--torso-theta N` | 24 | circumferential resolution of the torso surface |
| `--calibrate-metrication` | off | correct the graph's path-length bias |
| `--unbounded` | off | skip the torso boundary; comparison mode only |

### Atria (`--model mesh`)

| flag | default | meaning |
|---|---|---|
| `--no-atria` | off | ventricles only, so no P wave |
| `--atrial-cv CM_S` | 116 | isotropic atrial conduction velocity |
| `--calibrate-atrial-cv` | off | pick the velocity that gives a 95 ms P wave on *this* mesh |

### Rhythm strip (`--model mesh`)

| flag | default | meaning |
|---|---|---|
| `--rhythm` | off | multi-beat strip instead of a single cycle |
| `--heart-rate BPM` | 72 | heart rate |
| `--rhythm-duration S` | 10 | strip length in seconds |
| `--pr-interval MS` | 160 | atrioventricular delay (see note below) |
| `--rhythm-lead NAME` | `II` | lead drawn full-length along the figure's bottom |

`--pr-interval` is the delay from the **start of atrial activation** to the
**start of ventricular activation**. It stands in for the AV node, His bundle
and bundle branches, none of which exist as tissue in this mesh. It is close to
but not identical with a PR interval measured off a strip, because a measured PR
starts at the first *detectable* P deflection; at the default the simulator
reports 158 ms measured against 160 ms prescribed.

### Single-cell sweeps

| flag | default | meaning |
|---|---|---|
| `--ap-sweep` | off | full action potential, then scale I_Na and I_Kr separately |
| `--sweep-cell {endo,mid,epi}` | `epi` | cell type |
| `--sweep-levels N` | 10 | conductance levels, geometric about baseline |
| `--sweep-range X` | 4.0 | largest scale factor; smallest is its reciprocal |

### Ion-current readout

| flag | default | meaning |
|---|---|---|
| `--variation` | off | transmural table of I_Na / I_Kr peak and peak time |
| `--variation-bands N` | 9 | depth bands in that table |
| `--no-timeline` | off | suppress the long per-instant table |
| `--timeline-step MS` | adaptive | uniform readout stride |
| `--timeline-lead NAME` | `II` | lead shown in the readout's ECG column |

### Channel block (off by default)

| flag | default | meaning |
|---|---|---|
| `--ina-block PCT` | 0 | I_Na block, 0-100 |
| `--ikr-block PCT` | 0 | I_Kr block, 0-100 |

### Combinations that are refused

`--rhythm` requires `--model mesh`; `--rhythm` cannot be combined with
`--unbounded`; the atrial flags require `--model mesh`. Each is rejected with an
explanation rather than silently ignored.

---

## Outputs

### `--model mesh --rhythm`

| file | what it shows |
|---|---|
| `ecg12_rhythm_<bpm>bpm.png` | standard clinical layout: twelve leads in four 2.5 s columns, plus one lead full-length as a rhythm strip |
| `ecg_p_wave_detail.png` | two beats with the atrial and ventricular contributions drawn separately against their sum |
| `atrial_activation.png` | atrial activation time in three anatomical planes, with the sinoatrial node marked |

Without `--rhythm` the first file is named `ecg12_cycle_<bpm>bpm.png`. With
`--no-atria` you get the single-beat `ecg12_mesh.png` instead.

### `--model cable`

`ecg12_normal.png`, `track_current_maps.png`, `track_transmural.png`,
`track_ecg_correlation.png`, `ecg_ina_ikr_combined.png`. Adding `--ina-block` or
`--ikr-block` also writes a normal-versus-blocked 12-lead overlay and an action
potential comparison.

### `--ap-sweep`

`ap_complete.png` (the full single-cell action potential with its currents) and
`ap_sweep_ina_ikr.png` (how APD responds to scaling I_Na and I_Kr).

### Console output

Each mesh run prints, in order: mesh summary; activation solve with a
metrication check; template bank; beat summary; atrial mesh and activation;
anatomical frame; lead amplitudes; validation checks; a per-beat table of
measured intervals; and the I_Na / I_Kr readout.

The per-beat table is worth knowing about, because it is **measured back off the
finished waveform** by a QRS detector and a P-wave threshold, the same way you
would measure a real strip. It is not a restatement of the inputs:

```
   #   P onset    P peak     P end   P dur    P amp    QRS on      PR         R
   1       2.8      68.1      99.1    96.3    0.179     160.4   157.6     186.1
   2     836.1     901.4     932.4    96.3    0.179     993.7   157.6    1019.4
   ...
  12    9169.5    9234.8    9265.8    96.3    0.179    9327.1   157.6    9352.8
```

---

## Using it as a library

Everything the CLI does is available as functions. The package is not installed
by `requirements.txt`, so run your script **from the project root** (or put the
project root on `PYTHONPATH`) for `import ecg12sim` to resolve.

A full rhythm, from meshes to measured leads:

```python
from ecg12sim import (fetch_mesh, load_ventricular_mesh, load_atrial_mesh,
                      run_mesh_beat, run_atrial_beat, RhythmConfig,
                      run_rhythm_bem, check_p_wave, check_rhythm, format_checks)

vtk = fetch_mesh("01")                       # downloads on first use
vmesh = load_ventricular_mesh(vtk)           # tags 1 and 2
amesh = load_atrial_mesh(vtk)                # tags 3 and 4, same file

vbeat = run_mesh_beat(vmesh)                 # ventricular beat
abeat = run_atrial_beat(amesh)               # atrial beat, the P-wave source

cfg = RhythmConfig(heart_rate_bpm=72.0, duration_s=10.0, pr_interval_ms=160.0)
res = run_rhythm_bem(vbeat, abeat, cfg)      # the expensive step

print(res.summary())
print(format_checks(check_rhythm(res), "Rhythm"))
print(format_checks(check_p_wave(res), "P wave"))

t = res.t                                    # ms, 0.1 ms spacing
lead_ii = res.leads["II"]                    # mV
p_only = res.leads_atrial["II"]              # atrial contribution alone
```

`RhythmResult` carries the combined leads, each source's contribution
separately, the single-beat leads for detail plots, the anatomical frame, the
torso surface, and the P and atrial-T windows.

Smaller pieces:

```python
from ecg12sim import (solve_atrial_activation, sinoatrial_node,
                      calibrate_atrial_cv, build_atrial_template_bank,
                      measure_beats, p_wave_axis_deg, Drug)

node, dist_cm = sinoatrial_node(amesh)       # SAN at the SVC junction
cv = calibrate_atrial_cv(amesh)              # velocity for a 95 ms P wave
act = solve_atrial_activation(amesh, cv=cv)
bank = build_atrial_template_bank(drug=Drug.from_block_percent(0, 50))
```

Applying a channel block to both chambers:

```python
drug = Drug.from_block_percent(ina_block_pct=0, ikr_block_pct=70)
vbeat = run_mesh_beat(vmesh, drug=drug)
abeat = run_atrial_beat(amesh, drug=drug)
```

---

## How it works

### 1. Mesh, from tissue tags

The Rodero cohort labels every tetrahedron. Tag 1 is left-ventricular
myocardium, 2 right-ventricular, **3 left-atrial, 4 right-atrial**; tags 5-24 are
vessel walls, valve planes and vein inlets. `mesh.py` extracts the ventricles
and `atria.py` the atria, from the same file, so both are one subject's anatomy
in one coordinate system with no registration step.

**The atria and ventricles share 5,480 nodes** around the mitral and tricuspid
annuli. Solving activation over the union would let the atrial wavefront
depolarise ventricular myocardium directly, which is exactly what the annulus
fibrosus prevents in a real heart. So the two are solved as **separate meshes**,
and the only coupling is the prescribed AV delay. Within the atria, the left and
right chambers share 1,347 septal nodes, so interatrial conduction emerges from
tissue continuity rather than a hand-placed bridge.

### 2. Activation, by graph eikonal

Each mesh edge is weighted by `length / velocity` along its direction and a
single Dijkstra solve gives every node its activation time. Ventricular
conduction is **anisotropic**, built from the cohort's fibre and sheet fields,
and starts from distributed Purkinje root points. Atrial conduction is
**isotropic**, and that is forced by the data rather than chosen for
convenience: measured over all 397,373 atrial tetrahedra, the fibre orientation
tensor has eigenvalues (0, 0, 1), meaning every atrial fibre vector points in
one identical direction. It is a placeholder. The ventricular tags, by contrast,
give roughly (0.26, 0.28, 0.46), a genuinely dispersed field.

The sinoatrial node is placed as the right-atrial node nearest the superior vena
cava inlet centroid, read from tag 16. Deriving it from a tag the cohort
supplies makes it transfer between cases: 0.49 cm from that centroid on case 01
and 0.73 cm on case 02, with no per-case adjustment.

### 3. Reaction, by template bank

Running 450,000 coupled cells for a beat is not affordable, so the ionic model
is run a handful of times and the traces reused: a node at depth `rho` that
activates at time `AT` has `Vm(t) = template[rho](t - AT)`. The ventricles use
15 depth-indexed templates. The atria use **one**, because the cohort assigns no
universal coordinates outside the ventricles (every atrial node carries -10) and
the atrial wall is only 2.1 mm thick, two to three elements.

What makes a template atrial is `I_Kur`, the ultrarapid delayed rectifier that
distinguishes atrial from ventricular myocardium, plus a re-scaled conductance
set giving APD90 = 272 ms. `GKur` defaults to zero, so ventricular results are
numerically unchanged by its existence.

### 4. Forward model, bounded torso

The extracellular potential is linear in the nodal voltages, so each electrode's
signal is a fixed weighted sum over nodes — the lead field. Combined with the
template construction, every electrode signal becomes a convolution of templates
with activation-time histograms, computed exactly rather than sampled.

A parametric torso surface then bounds the medium and a boundary-element solve
gives true surface potentials. This matters: an insulated body cannot shed
current, which raises surface potentials well above free-space values. For a
centred dipole in a sphere the factor is exactly 3, and that is the analytic test
`torso.validate_sphere()` checks.

The atria get an **isotropic** intracellular conductivity, the trace-preserving
reduction of the ventricular tensor, `(σ_l + 2σ_t)/3 = 0.069 S/m`. Leaving it
anisotropic would orient all 397,373 atrial elements along the placeholder fibre
axis and inflate the P wave to 0.535 mV; the isotropic value gives 0.179 mV.

### 5. Rhythm, by superposition

The forward problem is linear in the source, so

```
φ_electrode(t) = φ_atrial(t) + φ_ventricular(t − PR)
```

is the exact solution, not an approximation, and a rhythm is a sum of shifted
copies of two single-beat solutions. This is why a 10-second strip costs the
same as one cycle. The boundary-element system is assembled and solved once for
both sources, and only the nine electrode vertices are kept before superposition
(602 vertices × 100,001 samples would be 480 MB; nine is 7 MB).

---

## What is validated, and how

Every mesh run prints its own checks. These are not smoke tests: lead polarity
and axis follow from where the electrodes sit relative to the heart, so they fail
if the geometry or the anatomical frame is wrong.

**Rhythm checks** — strip length, QRS count, mean RR, measured heart rate, and
whether beats overlap. All measured from the waveform by a slope-based QRS
detector.

**P-wave morphology** — a P before every QRS; P upright in I and II and inverted
in aVR; P axis in 0-75°; duration 80-120 ms; amplitude in II under 0.25 mV but
above 0.02 mV; PR in 120-200 ms; and the atrial T wave smaller than the P wave
and timed inside the QRS-T, where it hides clinically.

**Ventricular morphology** — aVR QRS area negative, II positive, precordial
R-wave progression, T upright in II and V5.

Measured at the defaults on cohort case 01:

| quantity | measured | normal |
|---|---|---|
| beats in 10 s / rate | 12 / **72.00 bpm** (RR 833.3 ms, sd 0.05) | — |
| P duration | 96 ms | 80-120 |
| P amplitude, lead II | 0.179 mV | ≤ 0.25 |
| P axis | +31.2° | 0-75 |
| PR interval | 158 ms measured, 160 prescribed | 120-200 |
| total atrial activation | 94.9 ms | 80-120 |
| atrial APD90 | 271.6 ms | 250-300 |
| atrial myocardial volume | 49.9 cm³ (0.22 × ventricular) | ratio ≈ 0.2 |
| QRS duration | 53.9 ms | 70-110 |
| QT interval | 322 ms | 350-440 |

15 of 15 rhythm and P-wave checks pass. The atrial conduction velocity was
calibrated on case 01 and then gave 93.0 ms on case 02 unchanged, so it is not
fitted to one mesh.

Two extra consistency probes are available as library calls. The forward model
has an analytic test — a dipole at the centre of an insulating sphere must give
surface potentials exactly three times the free-space value:

```bash
python -c "from ecg12sim import validate_sphere, format_sphere_check; \
print(format_sphere_check(validate_sphere()))"
# amplification vs unbounded : 3.0116  (exact: 3.0)
```

And `eikonal.measure_metrication()` quantifies how far graph-restricted paths
detour, since arrival times are biased late by paths that hop vertex to vertex
instead of cutting through tetrahedra (median inflation 1.08 on this mesh).

---

## Known limitations

These are real and worth knowing before trusting a number.

**The T wave is larger than the R wave** (0.97 mV against 0.52 mV in lead II,
where a real ECG has T at a third to an eighth of R). Ventricular repolarisation
in the mesh model is *prescribed* by the template bank's APD gradient rather than
emerging from electrotonic coupling between neighbours, which the 1-D cable does
get for free. This is the largest single departure from a clinical ECG and the
obvious next thing to fix. It is also why QRS detection is slope-based rather
than amplitude-based.

**Precordial R-wave progression is not monotonic** — the one ventricular
morphology check that fails, both with and without the atria.

**QRS is short (54 ms) and QT short (322 ms)** relative to clinical ranges. The
endocardial shell velocity of 600 cm/s, a Purkinje surrogate, is calibrated for
lead *morphology* rather than activation *duration*; at lower values the frontal
leads invert. That trade-off is documented in `eikonal.Velocities`.

**Atrial conduction is isotropic at 116 cm/s**, slightly above the 60-100 cm/s
measured in atrial working myocardium. Bachmann's bundle and the other
fibre-aligned fast pathways cannot be represented without an atrial fibre field,
so one velocity has to cover both. At 100 cm/s, the top of the measured range,
total atrial activation is 110 ms, still inside the normal P-wave window.

**The atrial action potential keeps a ventricular-like dome** (APD50/APD90 ≈
0.90 against a measured ≈ 0.55) because the shared `I_CaL` is a linear driving
term without calcium-dependent inactivation. Resting potential is pinned near
`E_K` = −86.9 mV, where human atrium rests at −75 to −80 mV, because ionic
concentrations are held fixed. Both affect atrial *repolarisation*, which is
clinically invisible anyway, not the P wave.

**Every beat in a rhythm is identical.** Superposition cannot capture
rate-dependent APD shortening (restitution), and the templates come from a
single stimulus rather than steady-state pacing. At 72 bpm there are 338 ms of
electrical silence between beats so they do not interact; above about 100 bpm
they overlap and the run says so (`diastolic margin` goes negative).

**The torso is homogeneous** — one conductivity, no separate lung, blood or
skeletal-muscle regions, and a parametric adult surface rather than a segmented
subject.

**The P wave carries low-amplitude fine structure**, 6-10% of its peak. This was
checked and is anatomical rather than numerical: the activation-time density
varies by about 57% as the wavefront crosses the atrial shell, and that figure
does not change when activation bins are coarsened 20-fold (0.563 → 0.579),
where counting noise would have fallen as the square root.

---

## Module map

| module | lines | role |
|---|---|---|
| `ionic_model.py` | 314 | human myocyte ionic model; I_Na, I_Kr, I_Kur explicit |
| `cable.py` | 160 | 1-D transmural strand and pseudo-ECG |
| `mesh.py` | 451 | cohort VTK parsing, ventricular extraction, caching, download |
| `atria.py` | 620 | **atrial extraction, sinoatrial node, atrial activation and template** |
| `eikonal.py` | 528 | anisotropic activation-time solve, Purkinje roots, metrication |
| `templates.py` | 420 | depth-indexed action-potential and current templates |
| `mesh_sim.py` | 410 | assembles a ventricular beat; exact spatial reductions |
| `forward.py` | 757 | anatomical frame, lead field, 12 leads, morphology checks |
| `torso.py` | 335 | bounded volume conductor, boundary elements, sphere validation |
| `rhythm.py` | 637 | **multi-beat assembly, QRS/P detection, rhythm and P checks** |
| `leads.py` | 121 | lead synthesis for the 1-D model |
| `simulate.py` | 80 | high-level orchestration for the cable model |
| `tracking.py` | 291 | I_Na / I_Kr analysis and transmural profiles |
| `timeline.py` | 451 | instant-by-instant current readout |
| `sweep.py` | 298 | single-cell conductance sweeps |
| `plotting.py` | 1399 | all figures |

Bold modules are the atrial and rhythm work.

Every module carries a long header docstring explaining not just what it does but
why it is built that way, including the measurements behind each calibrated
constant and the limitations each choice accepts. `atria.py` and `rhythm.py` are
the places to start for the P wave and the rhythm strip.

---

## References

Meshes are from the virtual cohort of adult healthy four-chamber heart meshes
derived from CT, licensed **CC-BY-4.0**:

> Rodero C, Strocchi M, Marciniak M, et al. (2021) Linking statistical shape
> models and simulated function in the healthy adult human heart. *PLOS
> Computational Biology* 17(4): e1008851. doi:10.1371/journal.pcbi.1008851

Mesh data: <https://doi.org/10.5281/zenodo.4590293> (record 4590294).

Ionic model gating follows the ten Tusscher & Panfilov human ventricular
formulations; `I_Kur` follows the Courtemanche, Ramirez & Nattel human atrial
formulation. The extracellular potential integral follows Gima & Rudy (2002),
and the boundary-element solid angles use the closed form of Van Oosterom &
Strackee (1983).

This is a research and teaching simulator. It is not a medical device and must not be used for clinical decisions.
