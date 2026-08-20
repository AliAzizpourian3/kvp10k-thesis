"""Collect immutable V5 training, selection, test, and SLURM provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from train_stage4b_v5 import utc_now, write_json


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _slurm_jobs(job_ids: list[str]) -> list[dict]:
    fields = "JobID,JobName,Partition,State,Elapsed,Start,End,NodeList,ExitCode"
    command = ["sacct", "-j", ",".join(job_ids), f"--format={fields}", "-P", "-X", "--noheader"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    keys = fields.split(",")
    return [dict(zip(keys, line.split("|"))) for line in result.stdout.splitlines() if line.strip()]


def _duration_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job_ids", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    configuration = _read(args.run_dir / "run_configuration.json")
    manifest = _read(args.run_dir / "run_manifest.json")
    completion = _read(args.run_dir / "training_completion.json")
    selection = _read(args.run_dir / "best_model" / "selection.json")
    test = _read(args.run_dir / "test_evaluation_official.json")
    jobs = _slurm_jobs(args.job_ids)
    write_json(
        args.output,
        {
            "collected_at": utc_now(),
            "completed_epochs": completion["completed_epoch"],
            "total_optimizer_steps": completion["global_optimizer_step"],
            "stopped_early": completion["stopped_early"],
            "selected_checkpoint": selection,
            "test_evaluation": {
                "evaluated_at": test["evaluated_at"],
                "checkpoint_path": test["checkpoint_path"],
                "checkpoint_sha256": test["checkpoint_sha256"],
                "selected_by_validation_only": test["selected_by_validation_only"],
            },
            "configuration": configuration,
            "data_counts": manifest["data_counts"],
            "initial_environment": manifest["environment"],
            "initial_command": manifest["command"],
            "resume_events": manifest.get("resume_events", []),
            "slurm_jobs": jobs,
            "total_allocated_seconds": sum(_duration_seconds(job["Elapsed"]) for job in jobs),
        },
    )


if __name__ == "__main__":
    main()
