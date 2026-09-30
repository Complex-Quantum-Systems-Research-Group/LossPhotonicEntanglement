# EP-MOKS file guide

Code for simulating two sequential polarization-qubit probes coupled to a spin chain.

| File | Purpose |
| --- | --- |
| `config.py` | Defines model settings, sweep grids, unit conversions, and output naming. |
| `pipeline.py` | Runs preflight validation, computes individual protocol points, and saves parameter sweeps. |
| `operators.py` | Provides basic matrices, tensor products, and operator embeddings. |
| `hamiltonians.py` | Builds spin Hamiltonians, probe magnetization operators, interaction generators, and commutators. |
| `states.py` | Constructs initial photon states and thermal spin states. |
| `new_evolution.py` | Implements unitary evolution and Kronecker-sum application routines. |
| `new_protocol.py` | Assembles probe profiles and the sequential photon-spin interaction protocol. |
| `observables.py` | Computes expectation values and partial traces. |
| `measures.py` | Computes entanglement, entropy, coherence, purity, mutual information, and fidelity diagnostics. |
| `validation.py` | Checks density-matrix physicality and operator unitarity. |
| `correlations.py` | Computes spin magnetization correlators and the magnetization-distance function using spectral methods. |
| `sector_correlations.py` | Computes magnetization-distance curves using conserved magnetization sectors. |
| `thermometry.py` | Builds cached temperature channels, QFI, apparatus CFI, and product/unrestricted input searches. |
| `thermometry_benchmark.py` | Runs validated local thermometry campaigns and independently refines Bell/product delays. |
| `thermometry_report.py` | Summarizes completed thermometry campaigns and plots QFI against common delay. |
| `thermometry_symmetry_check.py` | Checks the branch-complementation collapse of the QFI-optimal input and its origin in the h_z=0 sector-mirror symmetry, including the h_z!=0 breaking prediction. |
| `parity_preflight.py` | Evaluates the spin characteristic function and checks its relationships to photon-state diagnostics. |
| `channel_diagnostics.py` | Reconstructs the photon channel from matrix-unit responses and checks Choi positivity, trace preservation, Hermiticity, and unitality. |
| `check_convergence.py` | Compares selected protocol points across chain sizes and probe profiles, and provides point timing. |
| `finite_size_dm.py` | Runs sector-based finite-size campaigns, validates the sector calculation, and exports curves, extrema, and fits; also analyzes saved revivals and plots embedded reference arrays. |
| `compute_reduced_states.py` | Extracts and saves photon and spin reduced states from a saved joint density matrix. |
| `analyze_results.py` | Converts sweep summaries into tabulated diagnostics and summary reports. |
| `plot_results.py` | Generates sweep figures, extracts extrema, compares with perturbative predictions, and exports correlators. |
| `tests/test_physics.py` | Tests analytic controls, evolution implementations, channel reconstruction, correlators, and reporting behavior. |
| `requirements.txt` | Lists Python dependencies. |

## Run instructions

Run these commands in PowerShell from the repository root, using your Python environment.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set model settings, sweep grids, and the output root in `config.py`. The pipeline reads these settings; the CLI also accepts `--eta1`, `--eta2`, and `--n-spins` overrides. Use `python pipeline.py --help` to inspect supported options.

### Validate and run a sweep

```powershell
python pipeline.py validate
if ($LASTEXITCODE -ne 0) { throw "Validation failed" }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

python pipeline.py sweep
if ($LASTEXITCODE -ne 0) { throw "Sweep failed" }
```

Every sweep also runs mandatory preflight validation. Sweep outputs include reduced photon states and summary files.

To run a named campaign using the configured grid:

```powershell
python pipeline.py sweep --campaign my_campaign
```

For the built-in coarse weak-coupling campaign:

```powershell
python pipeline.py sweep --subgrid
```

### Analyze and plot saved output

Pass the intended summary explicitly when multiple campaigns exist. Replace the example path below with the saved summary JSON path:

```powershell
$summaryPath = "data/summary_example.json"
python analyze_results.py "$summaryPath" --output-dir "reports/my_campaign"
if ($LASTEXITCODE -ne 0) { throw "Analysis failed" }

python plot_results.py "$summaryPath"
if ($LASTEXITCODE -ne 0) { throw "Plotting failed" }
```

The scripts print their output locations. The plotter organizes figures by campaign relative to the summary location.

### Additional diagnostics

Run channel reconstruction or characteristic-function checks using their defaults:

```powershell
python channel_diagnostics.py
python parity_preflight.py
```

Inspect their `--help` output to select protocol points and other supported options.

Run finite-size magnetization-distance calculations, then analyze saved revival positions:

```powershell
python finite_size_dm.py
if ($LASTEXITCODE -ne 0) { throw "Finite-size calculation failed" }

python finite_size_dm.py --revivals
python finite_size_dm.py --plot-reference
```

`--revivals` expects the original N=8,10,12,14 and 6/500 K reference campaign keys; use `--curves PATH` to select its saved NPZ file. `--plot-reference` plots the original embedded arrays and does not automatically visualize a new campaign.

Inspect convergence and saved-state reduction commands:

```powershell
python check_convergence.py --help
python compute_reduced_states.py --help
```

Convergence comparisons use existing sweep output. Saved-state reduction requires a joint photon-spin density matrix, rather than the already reduced photon states saved by the sweep.

## Elliptical Kerr interaction

Both propagation paths implement `exp[-i (theta_k sigma_y + eta_k sigma_z) tensor M]`.
The defaults `eta1 = eta2 = 0` retain the rotation-only model. Nonzero eta is
lossless retardance, and is rejected for the exchange benchmark. Dense component
`G1`/`G2` include both angles for Kerr (`U = exp(-i G)`); the standalone generator
helper defaults to the legacy unit-angle generator.

```powershell
.\.venv\Scripts\python.exe ellipticity_preflight.py
.\.venv\Scripts\python.exe pipeline.py sweep --campaign ellipticity_smoke --n-spins 4 --n-temps 3 --n-delays 7 --eta1 0.25 --eta2 0.25
```

Preflight checks unitality for all angles, the manuscript's four elliptical
reference controls, and the original rotation-only controls at explicitly zero
ellipticity. Every zero-delay Phi+ Kerr point is checked against spectral
averaging of the exact Bell overlap, including unequal angles and zero coupling.

Summary rows save both eta angles, all four Bell populations, and theory
applicability. `p` and `Im_C` are supplied only for rotation-only Phi+ Kerr input;
they are null for elliptical output. `bell_infidelity = 1-F` remains available
for every model. The full reduced density matrices retain all coherences.
Nonzero-eta, nonzero-delay output is labeled exploratory: no elliptical
second-order delay law is assumed, and rotation-only perturbative plots are
skipped. The weak subgrid requires eta1=eta2=0.

Filenames contain both eta angles. Each sweep also saves a manifest containing
validation residuals, configuration source, actual model parameters, probe
weights, grids, units, git revision, and hashes of the Python source files.
Old summaries without eta fields are interpreted as rotation-only data.

Manuscript qualifications and research notes are kept locally in `manuscript/notes.md` (excluded from Git).

## Remaining manuscript computations

Run the full N=10 six-field follow-up locally using the existing production
kernel caches. This saves the midpoint comparison, full-simplex escape search,
and held-out rational fits under `reports/remaining_computations`:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe remaining_computations.py
if ($LASTEXITCODE -ne 0) { throw "Follow-up computation failed" }
.\.venv\Scripts\python.exe refine_remaining.py
if ($LASTEXITCODE -ne 0) { throw "Refinement audit failed" }
.\.venv\Scripts\python.exe summarize_remaining.py
if ($LASTEXITCODE -ne 0) { throw "Report failed" }
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The field files are checkpoints. Refinement checks endpoint-adjacent symmetric
maxima and independently audits each field's largest escape with new optimizer
seeds. The report distinguishes detection tolerance, negligible information,
and monotonicity failures. `symmetric_qfi.py` provides the exact two-block
cubic-over-quadratic QFI and quartic stationary-point candidates for nonsingular
parity-invariant kernels; singular cases use the spectral calculation.

## Model conventions

The probes are two polarization qubits. The spin bath is a finite XXZ spin-1/2
chain with S=sigma/2 and J>0 for antiferromagnetic exchange. KCuF3 motivates
the J approximately 34 meV scale; Delta=1 is an isotropic modeling approximation
that omits interchain exchange and residual anisotropy. A nonuniform Gaussian
probe is the default; collective magnetization provides the exact
delay-independent control. `config.py` defines the active production settings.

## Local files and Git

Keep manuscript text, drafts, and research notes in `manuscript/`. The consolidated
notes are in `manuscript/notes.md` on the local working copy. Generated data,
reports, figures, logs, environments, and local tool settings are ignored.
The README is the repository's file and workflow guide.

Run the thermometry benchmark and then summarize its saved campaigns with:

```powershell
python thermometry_benchmark.py
python thermometry_report.py
```

Use each command's `--help` for grid, output, and validation options.
