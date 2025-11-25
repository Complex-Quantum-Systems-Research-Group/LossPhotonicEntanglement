import numpy as np
from hamiltonians import build_total_hamiltonian
from states import bell_photons, thermal_spin_density_matrix
from evolution import evolve_density_matrix
from observables import photon_numbers, spin_at_site

# ============================================================================
# PARAMETERS
# ============================================================================

n_max = 2
N_spins = 8
omega1 = 1.0
omega2 = 1.5
g1 = 0.5
g2 = 0
J = -1.0
delta = 2.0
T_spin = 2.0

t_list = np.arange(0.0, 5.0, 0.05)

# ============================================================================
# BUILD SYSTEM
# ============================================================================

H = build_total_hamiltonian(n_max, N_spins, omega1, omega2, g1, g2, J, delta)

# Initial state: Bell photons ⊗ thermal spins
rho_photon = bell_photons(n_max, i=0, j=1)
rho_matter = thermal_spin_density_matrix(N_spins, J, delta, T=T_spin)
rho_initial = np.kron(rho_photon, rho_matter)

# ============================================================================
# TIME EVOLUTION
# ============================================================================

rho_t_list = evolve_density_matrix(H, rho_initial, t_list)

# ============================================================================
# SAVE DENSITY MATRICES
# ============================================================================

filename = f"rho_evolved_Nspins{N_spins}_nmax{n_max}_J{J}_delta{delta}_T{T_spin}.npz"
np.savez_compressed(filename, rho_t=rho_t_list, times=t_list)
print(f"Saved density matrices to: {filename}")

# ============================================================================
# CALCULATE OBSERVABLES
# ============================================================================

print("\nCalculating observables...")

# Photon observable: average photon number in mode 1
n1_avg = np.array([photon_numbers(rho, n_max, N_spins)[0] for rho in rho_t_list])

# Spin observable: z-component of first spin
sz_spin0 = np.array([spin_at_site(rho, 0, 'z', n_max, N_spins) for rho in rho_t_list])

print(f"\nPhoton mode 1: ⟨n₁⟩(t=0) = {n1_avg[0]:.4f}, ⟨n₁⟩(t=final) = {n1_avg[-1]:.4f}")
print(f"Spin 0: ⟨σ₀ᶻ⟩(t=0) = {sz_spin0[0]:.4f}, ⟨σ₀ᶻ⟩(t=final) = {sz_spin0[-1]:.4f}")

# Save observables
obs_filename = f"observables_Nspins{N_spins}_nmax{n_max}_J{J}_delta{delta}_T{T_spin}.npz"
np.savez(obs_filename, times=t_list, n1_avg=n1_avg, sz_spin0=sz_spin0)
print(f"\nSaved observables to: {obs_filename}")

# -------------------------
# SAVE FULL EVOLVED DENSITY MATRICES (PER PROTOCOL)
# -------------------------
# make a clear protocol name if not defined earlier
protocol_name = f"Nspins{N_spins}_nmax{n_max}_J{J}_delta{delta}_T{T_spin}"

rho_file = f"protocol_{protocol_name}_FULL_rho_t.npz"

# save compressed; keep complex dtype
np.savez_compressed(
    rho_file,
    times=t_list,
    rho_t=np.array(rho_t_list, dtype=complex)
)

print(f"[OK] Saved full protocol output → {rho_file}")
