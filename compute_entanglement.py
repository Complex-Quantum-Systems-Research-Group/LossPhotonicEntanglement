import numpy as np
from numpy.linalg import eigvalsh


# ============================================================
# VON NEUMANN ENTROPY
# ============================================================
def von_neumann_entropy(rho, tol=1e-12):
    rho = (rho + rho.conj().T) / 2  # enforce Hermiticity
    eigs = eigvalsh(rho)
    eigs = np.clip(eigs, 0, None)
    mask = eigs > 0
    return -np.sum(eigs[mask] * np.log(eigs[mask]))


# ============================================================
# MUTUAL INFORMATION
# ============================================================
def mutual_information(rho_total, rho_A, rho_B):
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_total)
    return S_A + S_B - S_AB


# ============================================================
# MAIN ENTANGLEMENT CALCULATOR
# ============================================================
def compute_entanglement(full_file, reduced_file):
    # Load full state
    full = np.load(full_file)
    rho_t = full["rho_t"]
    times = full["times"]

    # Load reduced states
    red = np.load(reduced_file)
    rho_p = red["rho_photon"]
    rho_s = red["rho_spin"]

    nt = len(times)

    S_photon = np.zeros(nt)
    S_spin   = np.zeros(nt)
    S_total  = np.zeros(nt)
    I_mutual = np.zeros(nt)

    for t in range(nt):
        S_photon[t] = von_neumann_entropy(rho_p[t])
        S_spin[t]   = von_neumann_entropy(rho_s[t])
        S_total[t]  = von_neumann_entropy(rho_t[t])
        I_mutual[t] = mutual_information(rho_t[t], rho_p[t], rho_s[t])

    # Save results
    out_file = full_file.replace("FULL_rho_t", "ENTANGLEMENT")

    np.savez(
        out_file,
        times=times,
        S_photon=S_photon,
        S_spin=S_spin,
        S_total=S_total,
        I_mutual=I_mutual
    )

    print(f"[OK] Saved entanglement data → {out_file}")


if __name__ == "__main__":
    import sys
    full_file    = sys.argv[1]
    reduced_file = sys.argv[2]
    compute_entanglement(full_file, reduced_file)
