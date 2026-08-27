# LossPhotonicEntanglement - corrected pre-results model

This revision changes the numerical model so that it matches the proposed observable: **polarization entanglement of two sequential probe photons**.  The previous code used two truncated bosonic modes and the Fock-state superposition `( |0,0> + |1,1> )/sqrt(2)`, which is not the polarization Bell pair described in the manuscript.

## Physical model

Hilbert-space ordering is

`photon_1 polarization qubit x photon_2 polarization qubit x spin_1 x ... x spin_N`.

The spin bath is a finite open XXZ chain

`H_s = J sum_i [Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Delta Sz_i Sz_{i+1}] - h_z sum_i Sz_i`,

with `S = sigma/2`, `hbar = k_B = 1`, and `J` the exchange-energy scale in this repo's sign convention: `J>0` is antiferromagnetic, `J<0` is ferromagnetic.

The primary EP-MOKS proxy is an impulsive, magnetization-conditioned polarization rotation

`U_k = exp[-i theta_k sigma_y^(photon k) tensor M_z^(probe)]`.

`M_z^(probe)` is a weighted local magnetization.  A nonuniform spatial profile is the default because a local MOKE spot samples local magnetization and, unlike the total `S_z`, it is not conserved by the XXZ exchange dynamics.  `probe_model="collective"` is retained as an exact control: because total `S_z` commutes with the XXZ Hamiltonian, the reduced two-photon state must be exactly independent of the inter-photon delay.

This is a **coherent effective Kerr-rotation model**, not a microscopic electronic theory of MOKE.  Absorption, polarization-dependent reflectivity, detector loss, and conditional/postselected channels are not included yet.

The production defaults use `theta1 = theta2 = 0.4` to produce a resolvable toy-model signal and scan dimensionless delays from 0 to 12 (about 232 fs for KCuF3), covering multiple finite-chain recurrence times. `weak_coupling_theta = 0.05` is reserved for perturbative validation. Optional campaign settings include `theta_nonperturbative = 1.0` and `theta_asym = (0.4, 0.2)`. These angles are not fitted material parameters.

## Material anchoring

The first material-anchored production pass targets **KCuF3**, a quasi-1D S=1/2 antiferromagnetic Heisenberg chain with intrachain exchange `J ≈ 34 meV` and near-isotropic exchange (~0.2% x-y anisotropy), Néel temperature `T_N = 39 K`.

Source: Lake, Tennant, Nagler et al., *"Longitudinal Magnetic Dynamics and Dimensional Crossover in the Quasi-One-Dimensional, Spin-1/2, Heisenberg Antiferromagnet KCuF3,"* arXiv:cond-mat/0503128.

`config.py` sets `J_meV = 34.0`, uses the antiferromagnetic sign convention (`J = +1.0` internally), and takes `Delta = 1.0` (the isotropic Heisenberg limit) as an **explicit modeling approximation**. Only the dominant intrachain XXZ term is represented — reported interchain exchange, the residual (~0.2%) anisotropy, and `T_N` are **not** encoded as active simulation parameters, since doing so would require extending the Hamiltonian to include interchain degrees of freedom, which this repository does not currently implement.

This calculation should therefore be read as a reduced one-dimensional proxy for dominant intrachain dynamics, not as a quantitatively complete microscopic model of KCuF3. `config.delay_fs()` converts a dimensionless sweep delay to femtoseconds via `hbar/J_meV`, for reporting only — the sweep itself remains dimensionless internally.

## Why the old Ising-Dicke/Tavis-Cummings framing was removed

The old `(a+a^dagger) sum sigma_z` coupling displaces a bosonic field and changes occupation number; it is not a polarization rotation.  The Tavis-Cummings interaction describes excitation exchange and is a cavity-QED benchmark, not a direct MOKE Hamiltonian.  An optional `exchange_benchmark` remains in `hamiltonians.py`/`new_protocol.py`, but manuscript claims must not identify it as MOKE.

## Diagnostics

The repository now distinguishes:

- `concurrence`: two-qubit entanglement measure;
- `mutual_information`: total correlation, not entanglement;
- photon-pair von Neumann entropy and purity: mixedness;
- `l1_coherence` and relative entropy of coherence: basis-dependent coherence measures;
- Bell-state fidelity.

The previous sum of squared off-diagonal elements was removed because it should not be presented as a standard resource-theoretic coherence monotone.

## Required pre-run sequence

```bash
python pipeline.py validate
pytest -q
```

Only after those pass should the full sweep be run:

```bash
python pipeline.py sweep
```

The production grid is 8 temperatures x 121 delays = 968 points (about 14.1 hours at 52.5 seconds per point). Do not duplicate it for perturbative validation. Instead run the coarse 2-temperature x 10-delay x 3-angle campaign:

```bash
python pipeline.py sweep --subgrid
```

Before interpreting a sweep, repeat selected points at larger `N_spins`, narrower grid spacing, and alternative probe profiles. A finite ten-spin chain does not exhibit a thermodynamic phase transition and cannot support claims of critical scaling.

Post-processing commands are:

```bash
python analyze_results.py
python plot_results.py
python channel_diagnostics.py --temperature-k 300 --delta-t 2.0
```

The channel diagnostic reconstructs all 16 matrix-unit responses before reporting Choi positivity, trace preservation, and unitality; a single Bell-state sweep is not sufficient for a channel-level claim. Both ordered magnetization correlators are exported by `plot_results.py` to `reports/`.

## Exact control identity

For an equal-weight collective probe,

`[H_XXZ, M_z^collective] = 0`.

The thermal state is a function of `H_XXZ`, and hence commutes with the conserved magnetization.  The two impulsive Kerr interactions are then conditioned on a static magnetization sector, so tracing out the spins gives a convex mixture of correlated photon-pair unitaries.  Consequently the reduced photon state is random-unitary and exactly independent of the free spin delay.  The test suite enforces this identity.

## Equal-coupling zero-delay invariance

For `theta1 = theta2 = theta` and `delta_t = 0`, the reduced photon state returns to the input Bell state for *any* probe profile, not only the conserved collective one. This follows because the Kerr rotation `exp(-i theta sigma_y)` is a real orthogonal matrix, so `(U(mu) ⊗ U(mu))|Phi+> = |Phi+>` on every eigenspace of the probe operator. `pipeline.py`'s `_validate_equal_coupling_zero_delay_invariance` and `test_physics.py`'s `test_equal_coupling_zero_delay_invariance_any_probe` enforce this identity across probe models and temperatures.

## Validation gate

A sweep is hard-gated by the mandatory preflight routine. `python pipeline.py sweep` calls `stage_validate()` before entering the parameter loop and aborts on any failed check. Analytic controls use four spins and test the Bell baseline, zero-coupling identity, collective/local commutators including the local formula's sign and prefactor, thermal-state physicality, `U1`/`Udelay`/`U2` unitarity, nontrivial multi-sector collective-probe delay independence, equal-coupling zero-delay invariance, and representative configured smoke points. A final point uses the configured production size (`N=10`). The independent `test_physics.py` suite also checks analytic limits and the production fast path against a brute-force dense implementation.
