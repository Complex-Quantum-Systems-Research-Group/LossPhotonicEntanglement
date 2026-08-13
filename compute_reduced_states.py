"""Utility for reducing saved full density matrices from the corrected model."""
from __future__ import annotations

import argparse
import numpy as np

from observables import partial_trace_photons, partial_trace_spins
from validation import assert_density_matrix


def compute_reduced_states(rho_file: str, n_spins: int, output_file: str | None = None) -> str:
    rho = np.load(rho_file)
    assert_density_matrix(rho)
    rho_photons = partial_trace_spins(rho, n_spins)
    rho_spins = partial_trace_photons(rho, n_spins)
    assert_density_matrix(rho_photons)
    assert_density_matrix(rho_spins)
    if output_file is None:
        output_file = rho_file.rsplit(".", 1)[0] + "_reduced.npz"
    np.savez_compressed(output_file, rho_photons=rho_photons, rho_spins=rho_spins)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rho_file")
    parser.add_argument("n_spins", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(compute_reduced_states(args.rho_file, args.n_spins, args.output))
