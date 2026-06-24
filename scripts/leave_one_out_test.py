import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def update_evaluator_restore_hint(yaml_text: str, restore_hint: str) -> str:
    lines = yaml_text.splitlines()
    out_lines = []
    in_eval = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if indent == 0 and stripped.startswith("evaluator_cfg:"):
            in_eval = True
            out_lines.append(line)
            continue

        if in_eval and indent == 0 and stripped and not stripped.startswith("#"):
            in_eval = False

        if in_eval and stripped.startswith("restore_hint:"):
            out_lines.append(f"{' ' * indent}restore_hint: {restore_hint}")
            continue

        out_lines.append(line)

    trailing_newline = "\n" if yaml_text.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline


def subject_to_suffix(subject: str) -> str:
    head = subject.split("_")[0]
    return head[1:]


def build_default_subjects() -> list[str]:
    return [f"S{idx:02d}_T01" for idx in range(1,31)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automate leave-one-subject testing for OpenGait.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to the OpenGait repo root (defaults to script parent).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of subjects processed (for quick checks).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue even if a test run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended actions without modifying files or running tests.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    dataset_file = repo_root / "datasets" / "Anonymous" / "train_test.json"
    config_file = repo_root / "configs" / "gaitset" / "anonymous_test.yaml"
    log_dir = repo_root / "logs" / "leave_one_out_test"

    if not dataset_file.exists():
        print(f"Missing dataset file: {dataset_file}")
        return 1
    if not config_file.exists():
        print(f"Missing config file: {config_file}")
        return 1

    dataset_payload = load_json(dataset_file)
    test_set = dataset_payload.get("TEST_SET", [])
    subjects = build_default_subjects()

    if set(subjects) != set(dataset_payload.get("TRAIN_SET", [])):
        print("Warning: TRAIN_SET does not match S01_T01..S30_T01 exactly.")

    if args.limit:
        subjects = subjects[: args.limit]

    yaml_original = config_file.read_text(encoding="utf-8")
    dataset_original = json.dumps(dataset_payload, indent=4)

    if args.dry_run:
        print("Dry run enabled: no files will be modified and no tests will start.")

    #run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for subject in subjects:
            suffix = subject_to_suffix(subject)
            save_name = f"GaitSet_Finetune_{suffix}"
            restore_hint = (
                f"./output/MyClassification/GaitSet/{save_name}/checkpoints/"
                f"{save_name}-06000.pt"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{save_name}.log"

            print(f"\n=== Test subject {subject} -> {save_name} ===")
            print(f"Test set: [{subject}]")
            print(f"Restore hint: {restore_hint}")
            print(f"Log file: {log_path}")

            if args.dry_run:
                continue

            dataset_payload["TEST_SET"] = [subject]
            write_json(dataset_file, dataset_payload)

            updated_yaml = update_evaluator_restore_hint(yaml_original, restore_hint)
            config_file.write_text(updated_yaml, encoding="utf-8")

            command = (
                "python -m torch.distributed.launch --nproc_per_node=1 "
                "opengait/main.py --cfgs ./configs/gaitset/Anonymous_test.yaml --phase test"
            )

            with log_path.open("w", encoding="utf-8") as log_handle:
                result = subprocess.run(
                    command,
                    cwd=str(repo_root),
                    shell=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )

            if result.returncode != 0:
                print(f"Run failed for {subject} (code {result.returncode}).")
                if not args.continue_on_error:
                    return result.returncode

    finally:
        if not args.dry_run:
            dataset_file.write_text(dataset_original + "\n", encoding="utf-8")
            config_file.write_text(yaml_original, encoding="utf-8")

    print("\nAll test runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
