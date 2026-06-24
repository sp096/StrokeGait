import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def update_trainer_save_name(yaml_text: str, save_name: str) -> str:
    lines = yaml_text.splitlines()
    out_lines = []
    in_trainer = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if indent == 0 and stripped.startswith("trainer_cfg:"):
            in_trainer = True
            out_lines.append(line)
            continue

        if in_trainer and indent == 0 and stripped and not stripped.startswith("#"):
            in_trainer = False

        if in_trainer and stripped.startswith("save_name:"):
            out_lines.append(f"{' ' * indent}save_name: {save_name}")
            continue

        out_lines.append(line)

    trailing_newline = "\n" if yaml_text.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline


def subject_to_suffix(subject: str) -> str:
    head = subject.split("_")[0]
    return head[1:]


def build_default_subjects() -> list[str]:
    return [f"S{idx:02d}_T01" for idx in range(1,31)]


def notify_completion(dry_run: bool) -> None:
    if dry_run:
        print("Dry run complete. No training was started.")
        return

    if winsound is not None:
        # 3 short beeps to signal completion.
        for _ in range(3):
            winsound.Beep(1000, 500)
    else:
        print("All runs complete (notification beep unavailable).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automate leave-one-subject training for OpenGait.",
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
        help="Continue even if a training run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended actions without modifying files or running training.",
    )
    parser.add_argument(
        "--test-beep",
        action="store_true",
        help="Play the completion notification and exit.",
    )
    args = parser.parse_args()

    if args.test_beep:
        notify_completion(False)
        return 0

    repo_root = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    dataset_file = repo_root / "datasets" / "Anonymous" / "train_test.json"
    config_file = repo_root / "configs" / "gaitset" / "anonymous_test.yaml"
    log_dir = repo_root / "logs" / "leave_one_out"

    if not dataset_file.exists():
        print(f"Missing dataset file: {dataset_file}")
        return 1
    if not config_file.exists():
        print(f"Missing config file: {config_file}")
        return 1

    dataset_payload = load_json(dataset_file)
    train_set = dataset_payload.get("TRAIN_SET", [])
    subjects = build_default_subjects()

    #if set(subjects) != set(train_set):
    #    print("Warning: TRAIN_SET does not match S01_T01..S30_T01 exactly.")
    #    subjects = list(train_set)

    if args.limit:
        subjects = subjects[: args.limit]

    yaml_original = config_file.read_text(encoding="utf-8")
    dataset_original = json.dumps(dataset_payload, indent=4)

    if args.dry_run:
        print("Dry run enabled: no files will be modified and no training will start.")

    try:
        for subject in subjects:
            if subject not in train_set:
                print(f"Skipping {subject}: not present in TRAIN_SET.")
                continue

            new_train = [item for item in train_set if item != subject]
            suffix = subject_to_suffix(subject)
            save_name = f"Gaitset_Finetune_{suffix}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{save_name}.log"

            print(f"\n=== Leave out {subject} -> {save_name} ===")
            print(f"Train set size: {len(new_train)}")
            print(f"Log file: {log_path}")

            if args.dry_run:
                continue

            dataset_payload["TRAIN_SET"] = new_train
            write_json(dataset_file, dataset_payload)

            updated_yaml = update_trainer_save_name(yaml_original, save_name)
            config_file.write_text(updated_yaml, encoding="utf-8")

            command = (
                "python -m torch.distributed.launch --nproc_per_node=1 "
                "opengait/main.py --cfgs ./configs/gaitset/Anonymous_test.yaml --phase train"
            )

            with log_path.open("w", encoding="utf-8") as log_handle:
                # subprocess.run is blocking, so each run finishes before the next starts.
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

    print("\nAll runs complete.")
    notify_completion(args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
