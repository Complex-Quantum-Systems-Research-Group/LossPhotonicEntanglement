import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import config as cfg
from pipeline import run_point

SOURCE = Path(
    "reports/ellipticity_wiring_audit/interpolated_extrema.csv"
)
OUTPUT = SOURCE.with_name("refined_extrema.csv")

N_SPINS = 8
THETA = 0.4
WINDOW_FS = 8.0
XATOL_FS = 1e-3


@lru_cache(maxsize=4096)
def fidelity(T_kelvin, theta, eta, delay_fs):
    """Evaluate the full propagated Bell fidelity."""
    temperature = T_kelvin * cfg.K_B_MEV_K / cfg.J_meV
    delay = delay_fs / float(cfg.delay_fs(1.0))

    result = run_point(
        T=temperature,
        delta_t=delay,
        n_spins=N_SPINS,
        theta1=theta,
        theta2=theta,
        eta1=eta,
        eta2=eta,
    )
    return float(result["bell_fidelity"])


def refine_extremum(
    run_fn, T, theta, eta, t_grid, kind, window_fs=WINDOW_FS
):
    """T in Kelvin; all delay arguments and outputs in femtoseconds."""
    signs = {"min": 1, "minimum": 1, "max": -1, "maximum": -1}
    if kind not in signs:
        raise ValueError(f"Unknown extremum kind: {kind!r}")
    if window_fs <= 0:
        raise ValueError("window_fs must be positive")

    sign = signs[kind]
    lo = max(0.0, t_grid - window_fs)
    hi = t_grid + window_fs

    def objective(t):
        value = sign * run_fn(T, theta, eta, float(t))
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite fidelity at {t} fs")
        return value

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": XATOL_FS, "maxiter": 100},
    )
    if not result.success:
        raise RuntimeError(result.message)

    # Reject boundary solutions and verify local extremum character.
    h = 0.05  # fs
    if not lo + h < result.x < hi - h:
        raise RuntimeError(
            f"Optimum near window boundary at T={T}, eta={eta}: "
            f"{result.x:.6f} fs. Inspect the search window."
        )

    center = objective(result.x)
    if not (
        center < objective(result.x - h)
        and center < objective(result.x + h)
        and center < objective(lo)
        and center < objective(hi)
    ):
        raise RuntimeError(
            f"Could not verify an isolated {kind} near {t_grid} fs "
            f"at T={T}, eta={eta}."
        )

    return float(result.x), float(sign * result.fun)


def main():
    if cfg.interaction_type != "kerr" or cfg.bell_state != "phi_plus":
        raise ValueError("This comparison requires Kerr interaction and Phi+ input.")

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    columns = [
        "T_kelvin", "kind", "grid_fs",
        "rotation_refined_fs", "elliptical_refined_fs", "shift_fs",
        "rotation_fidelity", "elliptical_fidelity",
        "window_fs", "optimizer_xatol_fs",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Save each completed pair immediately.
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            T = float(row["T_kelvin"])
            grid = float(row["grid_fs"])
            kind = row["kind"].strip().lower()

            print(
                f"[{index}/{len(rows)}] T={T:.0f} K, "
                f"{kind}, coarse delay={grid:.3f} fs",
                flush=True,
            )

            t0, f0 = refine_extremum(fidelity, T, THETA, 0.0, grid, kind)
            t1, f1 = refine_extremum(fidelity, T, THETA, 0.25, grid, kind)

            writer.writerow({
                "T_kelvin": T,
                "kind": kind,
                "grid_fs": grid,
                "rotation_refined_fs": t0,
                "elliptical_refined_fs": t1,
                "shift_fs": t1 - t0,
                "rotation_fidelity": f0,
                "elliptical_fidelity": f1,
                "window_fs": WINDOW_FS,
                "optimizer_xatol_fs": XATOL_FS,
            })
            handle.flush()
            print(
                f"  rotation={t0:.6f}, elliptical={t1:.6f}, "
                f"shift={t1-t0:+.6f} fs",
                flush=True,
            )

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()