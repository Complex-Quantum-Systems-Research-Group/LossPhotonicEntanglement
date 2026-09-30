"""Create compact, machine-readable tables from an existing EP-MOKS sweep.

This is post-processing only: it never runs a simulation. By default it finds
the configured ``data/summary_*.json`` and writes ignored artifacts to
``reports/``.

Usage:
    python analyze_results.py
    python analyze_results.py path/to/summary.json --output-dir reports
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import config as cfg


POINT_COLUMNS = (
    "eta1", "eta2", "theory_status", "p", "Im_C", "bell_infidelity",
    "T",
    "T_kelvin",
    "delta_t",
    "delta_t_fs",
    "concurrence",
    "bell_fidelity",
    "purity",
    "entropy_bits",
    "mutual_information_bits",
    "l1_coherence",
    "relative_entropy_coherence_bits",
)


def _load_summary(path: str | None) -> tuple[list[dict], Path]:
    if path is None:
        data_dir = Path(cfg.output_root)
        expected = data_dir / f"summary_{cfg.filename_tag()}.json"
        candidates = [expected] if expected.exists() else sorted(data_dir.glob("summary_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No summary_*.json found in {data_dir}")
        source = candidates[0]
    else:
        source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return json.load(handle), source


def _group_by_temperature(rows: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["T_kelvin"], []).append(row)
    for temperature in groups:
        groups[temperature].sort(key=lambda row: row["delta_t_fs"])
    return groups


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temperature_summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for temperature, group in sorted(_group_by_temperature(rows).items()):
        first = group[0]
        last = group[-1]
        minimum = min(group, key=lambda row: row["concurrence"])
        summaries.append(
            {
                "T_kelvin": temperature,
                "n_delays": len(group),
                "max_delay_fs": last["delta_t_fs"],
                "initial_concurrence": first["concurrence"],
                "terminal_concurrence": last["concurrence"],
                "minimum_concurrence": minimum["concurrence"],
                "minimum_concurrence_delay_fs": minimum["delta_t_fs"],
                "maximum_concurrence_loss": first["concurrence"] - minimum["concurrence"],
                "terminal_bell_fidelity": last["bell_fidelity"],
                "terminal_purity": last["purity"],
            }
        )
    return summaries


def analyze(rows: list[dict], source: Path) -> tuple[list[dict], dict]:
    by_temperature = _temperature_summaries(rows)
    minimum = min(rows, key=lambda row: row["concurrence"])
    maximum_loss = max(by_temperature, key=lambda row: row["maximum_concurrence_loss"])
    temperatures = sorted({row["T_kelvin"] for row in rows})
    delays = sorted({row["delta_t_fs"] for row in rows})
    headline = {
        "eta1": rows[0].get("eta1", 0.0),
        "eta2": rows[0].get("eta2", 0.0),
        "theory_status": sorted({r.get("theory_status", "legacy_rotation_only") for r in rows}),
        "source": str(source),
        "n_points": len(rows),
        "n_temperatures": len(temperatures),
        "n_delays": len(delays),
        "temperature_range_kelvin": [temperatures[0], temperatures[-1]],
        "delay_range_fs": [delays[0], delays[-1]],
        "global_minimum_concurrence": minimum["concurrence"],
        "global_minimum_at": {
            "T_kelvin": minimum["T_kelvin"],
            "delta_t_fs": minimum["delta_t_fs"],
        },
        "largest_concurrence_loss": maximum_loss["maximum_concurrence_loss"],
        "largest_loss_temperature_kelvin": maximum_loss["T_kelvin"],
        "campaign": rows[0].get("campaign"),
        "interpretation_note": (
            "theta is an effective toy-model coupling unless independently calibrated; "
            "do not interpret this sweep as a quantitative MOKE prediction."
        ),
    }
    return by_temperature, headline


def _report_weak_coupling_gate(rows: list[dict]) -> None:
    """Report applicability of the perturbative residual gate without failing."""
    first = rows[0]
    if any(r.get("eta1", 0.0) != 0.0 or r.get("eta2", 0.0) != 0.0 for r in rows):
        print("SKIP weak-coupling gate: elliptical delay response is exploratory")
        return
    theta1 = float(first.get("theta1", cfg.theta1))
    theta2 = float(first.get("theta2", cfg.theta2))
    if not (
        abs(theta1 - cfg.weak_coupling_theta) <= 1e-12
        and abs(theta2 - cfg.weak_coupling_theta) <= 1e-12
    ):
        print(
            f"SKIP weak-coupling gate: theta1={theta1:.3g}, theta2={theta2:.3g}; "
            f"gate applies only to theta={cfg.weak_coupling_theta:.3g}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", help="Existing summary JSON; auto-detected if omitted")
    parser.add_argument("--output-dir", default="reports", help="Destination directory (default: reports)")
    args = parser.parse_args()

    rows, source = _load_summary(args.summary)
    if not rows:
        raise ValueError(f"Sweep summary is empty: {source}")
    _report_weak_coupling_gate(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_temperature, headline = analyze(rows, source)

    point_path = output_dir / "sweep_points.csv"
    temperature_path = output_dir / "temperature_summary.csv"
    headline_path = output_dir / "headline_summary.json"
    _write_csv(point_path, rows, POINT_COLUMNS)
    _write_csv(temperature_path, by_temperature, tuple(by_temperature[0]))
    with headline_path.open("w", encoding="utf-8") as handle:
        json.dump(headline, handle, indent=2)
        handle.write("\n")

    print(f"Loaded {len(rows)} existing points from {source}")
    print(f"Minimum concurrence: {headline['global_minimum_concurrence']:.9f}")
    print(f"Largest concurrence loss: {headline['largest_concurrence_loss']:.9f}")
    print(f"Wrote {point_path}")
    print(f"Wrote {temperature_path}")
    print(f"Wrote {headline_path}")


if __name__ == "__main__":
    main()
