import numpy as np
from observables import partial_trace_spins, partial_trace_photons


def compute_reduced_states(rho_file, n_max, N_spins):
    # Load the full density matrices
    data = np.load(rho_file)
    rho_t = data["rho_t"]
    times = data["times"]

    nt = len(rho_t)

    # Dimensions
    dim_p = n_max * n_max
    dim_s = 2**N_spins

    rho_photon = np.zeros((nt, dim_p, dim_p), dtype=complex)
    rho_spin   = np.zeros((nt, dim_s, dim_s), dtype=complex)

    # Compute reduced states
    for t in range(nt):
        rho_photon[t] = partial_trace_spins(rho_t[t], n_max, N_spins)
        rho_spin[t]   = partial_trace_photons(rho_t[t], n_max, N_spins)

    # Save results
    out_file = rho_file.replace("FULL_rho_t", "REDUCED_rho")

    np.savez(
        out_file,
        times=times,
        rho_photon=rho_photon,
        rho_spin=rho_spin
    )

    print(f"[OK] Saved reduced states → {out_file}")


if __name__ == "__main__":
    import sys
    rho_file = sys.argv[1]
    n_max    = int(sys.argv[2])
    N_spins  = int(sys.argv[3])

    compute_reduced_states(rho_file, n_max, N_spins)
