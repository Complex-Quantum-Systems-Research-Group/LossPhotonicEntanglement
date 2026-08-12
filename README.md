# LM-interaction

Hybrid quantum system simulation: two photon modes coupled to an interacting spin chain, evolved through a 4-step unitary protocol, then analyzed for entanglement and coherence across a temperature / delay-time parameter sweep.

Hilbert space ordering: `mode1 x mode2 x spin1 x ... x spinN`, dimension `n_max^2 x 2^N`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python pipeline.py evolve       # step 1: sweep, save full evolved rho
python pipeline.py photon       # step 2: trace out spins -> data_photon/
python pipeline.py observable   # step 3: von Neumann entropy + mutual info
python pipeline.py coherence    # step 3 (alt): coherence measures, own sweep
```

Then run the matching `.ipynb` notebook (not included in this cleanup pass) to plot results.

All sweep parameters (`n_max`, `N_spins`, `omega1/2`, `g1/2`, `J`, `delta`, `interaction_type`, `tau_1/2`, `final_evolution_time`, `temperature_list`, `delta_t_list`) live in `config.py`. Edit there, not in `pipeline.py`.

**Known open item:** `pipeline.py`'s `coherence` stage currently uses `g1=g2` from `config.py`; an earlier note in this repo referenced `g=20` for that stage specifically. Confirm which is correct before running.

## Modules

| File | Purpose |
|---|---|
| `operators.py` | Shared building blocks: Pauli matrices, bosonic ladder operators, `operator_at_spin_site`. |
| `hamiltonians.py` | Bosonic, XXZ spin, and spin-boson coupling Hamiltonians (Tavis-Cummings or Ising-Dicke); spin-only Hamiltonian for thermal states. |
| `observables.py` | Expectation values, partial traces (photon/spin), photon number operators, energy and energy variance. |
| `states.py` | Bell photon states, product Fock states, thermal spin density matrices. |
| `new_evolution.py` | Unitary evolution `rho(t) = U(t) rho_0 U^dagger(t)`; exposes `unitary_from_hamiltonian(H, t)`. |
| `new_protocol.py` | Hamiltonian-agnostic 4-step pipeline via `apply_unitaries(rho, [U1, ..., Un])`. |
| `measures.py` | Von Neumann entropy, mutual information, off-diagonal coherence, relative entropy of coherence, Kraus channels. |
| `compute_reduced_states.py` | CLI: loads a full `rho(t)` array, computes photon and spin reduced states at each time step, saves as `.npz`. |
| `pipeline.py` | Entry point for the four sweep stages (see Run, above). |
| `config.py` | Single source of truth for all sweep/model parameters. |

## Data flow

```
pipeline.py evolve      -> ising_dickie/ or tavis_cummings/   (full rho)
pipeline.py photon      -> data_photon/                       (photon-reduced rho)
pipeline.py observable  -> data_observable/                   (entropy, mutual info)
pipeline.py coherence   -> data_coherence/                    (rho + coherence measures)
```

## Notes on this cleanup pass

- Removed `evolution.py`, `protocol.py` (dead code, superseded by `new_evolution.py`/`new_protocol.py`).
- Removed a stray module-level call in `states.py` (`product_photons(4, ...)` executed on import, discarded).
- Deduplicated `create_pauli_matrices`, `create_bosonic_operators`, `operator_at_spin_site` out of `hamiltonians.py`/`observables.py` into `operators.py`.
- Merged `coherence.py` + `entanglement.py` -> `measures.py`.
- Merged `run.py`, `run_rho_photon.py`, `run_observable.py`, `run_coherence.py` -> `pipeline.py`.
- Centralized sweep parameters into `config.py`.
- Not checked/fixed: `.gitignore` content, notebook hardcoded Windows paths, duplicate files/folders outside this upload (`.git - Copy/`, `compute_reduced_states - Copy.py`).
