"""Reconstruct and validate the two-photon channel at one protocol point.

The sweep's Bell-state output is not process tomography. This module applies
the protocol to all 16 matrix units of the four-dimensional photon space,
constructs the Choi matrix, and reports CP, TP, Hermiticity-preservation, and
unitality residuals.

Usage:
    python channel_diagnostics.py --temperature-k 300 --delta-t 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
from new_protocol import full_pipeline_unitary
from observables import partial_trace_spins
from validation import assert_density_matrix


CHOI_CONVENTION = (
    "input-major Kronecker stacking: "
    "J[(i,a),(j,b)] = Lambda(|i><j|)[a,b] = "
    "sum_ij |i><j| tensor Lambda(|i><j|); Tr_out(J) = I"
)


def apply_photon_channel(
    photon_operator, *, n_spins, J, delta, temperature, delta_t,
    theta1, theta2, probe_model="local_gaussian", probe_sigma_sites=1.0,
    h_z=0.0, periodic=False,
):
    """Apply the primary Kerr protocol linearly to an arbitrary 4x4 operator."""
    photon_operator = np.asarray(photon_operator, dtype=complex)
    if photon_operator.shape != (4, 4):
        raise ValueError("photon_operator must have shape (4, 4)")
    evolved = full_pipeline_unitary(
        n_spins=n_spins, J=J, delta=delta, temperature=temperature,
        delta_t=delta_t, theta1=theta1, theta2=theta2,
        probe_model=probe_model, probe_sigma_sites=probe_sigma_sites,
        h_z=h_z, periodic=periodic, interaction_type="kerr",
        photon_operator=photon_operator,
    )
    return partial_trace_spins(evolved, n_spins)


def reconstruct_choi(**protocol) -> np.ndarray:
    """Construct ``J(E)=sum_ij |i><j| tensor E(|i><j|)`` for d=4."""
    dimension = 4
    choi = np.zeros((dimension**2, dimension**2), dtype=complex)
    for i in range(dimension):
        for j in range(dimension):
            matrix_unit = np.zeros((dimension, dimension), dtype=complex)
            matrix_unit[i, j] = 1.0
            output = apply_photon_channel(matrix_unit, **protocol)
            choi += np.kron(matrix_unit, output)
    return choi


def channel_residuals(choi: np.ndarray) -> dict:
    """Return numerical CPTP, Hermiticity, and unitality diagnostics."""
    dimension = 4
    tensor = choi.reshape(dimension, dimension, dimension, dimension)
    trace_output = np.einsum("iaja->ij", tensor)
    # E(I) is the trace over the input indices of the Choi matrix.
    image_identity = np.einsum("iaib->ab", tensor)
    hermiticity_residual = float(np.linalg.norm(choi - choi.conj().T))
    hermitian_choi = 0.5 * (choi + choi.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian_choi)
    return {
        "choi_min_eigenvalue": float(eigenvalues.min()),
        "choi_trace": float(np.trace(choi).real),
        "hermiticity_residual": hermiticity_residual,
        "trace_preservation_residual": float(np.linalg.norm(trace_output - np.eye(dimension))),
        "unitality_residual": float(np.linalg.norm(image_identity - np.eye(dimension))),
    }


def validate_choi_reconstruction(choi: np.ndarray, atol=1e-10) -> dict:
    """Validate normalized Choi positivity and report implementation residuals.

    Unitality is Proposition 5's implementation check. It is deliberately not
    part of ``cptp_pass``: a nonzero value indicates a reconstruction bug, not
    a physical result of this configured channel.
    """
    dimension = 4
    normalized_choi = choi / dimension
    assert_density_matrix(normalized_choi, atol=atol)
    residuals = channel_residuals(choi)
    cptp_pass = (
        residuals["choi_min_eigenvalue"] >= -atol
        and residuals["hermiticity_residual"] <= atol
        and residuals["trace_preservation_residual"] <= atol
    )
    return {
        "choi_convention": CHOI_CONVENTION,
        "choi_normalization_for_positivity": "J / 4",
        "cptp_pass": cptp_pass,
        "validation": {
            **residuals,
            "unitality_label": (
                "Proposition 5 implementation check; nonzero indicates a "
                "channel-reconstruction bug, not physics"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--delta-t", type=float, default=2.0, help="Dimensionless delay in hbar/J")
    parser.add_argument("--output", default="reports/channel_diagnostics.json")
    args = parser.parse_args()
    temperature = args.temperature_k * cfg.K_B_MEV_K / cfg.J_meV
    protocol = dict(
        n_spins=cfg.N_spins, J=cfg.J, delta=cfg.delta, temperature=temperature,
        delta_t=args.delta_t, theta1=cfg.theta1, theta2=cfg.theta2,
        probe_model=cfg.probe_model, probe_sigma_sites=cfg.probe_sigma_sites,
        h_z=cfg.h_z, periodic=cfg.periodic,
    )
    diagnostics = validate_choi_reconstruction(reconstruct_choi(**protocol))
    diagnostics["parameters"] = {
        "temperature_kelvin": args.temperature_k,
        "delta_t": args.delta_t,
        "n_spins": cfg.N_spins,
        "theta1": cfg.theta1,
        "theta2": cfg.theta2,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
