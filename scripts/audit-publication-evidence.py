"""Read-only study audit. Writes only the separate publication report directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    return path


def audit(study, root=ROOT):
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    def read(name):
        path = study / name
        if not path.exists():
            check(name, False, "Missing; no completion inferred")
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not value:
                raise ValueError("Expected a nonempty JSON object")
            return value
        except (ValueError, OSError) as error:
            check(name, False, str(error))
            return None

    def match(name, path, expected):
        check(name, path.is_file() and digest(path) == expected, str(path))

    protocol = read("protocol.json")
    manifest = read("split-manifest.json")
    source = read("source-manifest.json")
    if protocol and manifest:
        match("protocol fingerprint", study / "protocol.json", manifest.get("protocol_sha256"))
        match("supplied DDPF fingerprint", root / "models/balanced_ddpf.pt", manifest.get("ddpf_sha256"))
        check("full training protocol", protocol.get("strfe_epochs") == 40 and protocol.get("graph_epochs") == 40 and protocol.get("min_epochs") == 12 and len(protocol.get("seeds", [])) == 3, "Frozen 40-epoch budgets, 12-epoch minimum and three seeds")
        partitions = manifest.get("partitions", {})
        sets = [set(partitions.get(key, [])) for key in ("train", "val", "test")]
        check("sequence separation", all(not sets[i] & sets[j] for i in range(3) for j in range(i)), "Train/validation/test sequence IDs must not overlap")
        check("split sizes", list(map(len, sets)) == [68, 22, 22], "Expected 68/22/22 unique sequences")
        check("development exposure excluded", not (sets[1] | sets[2]) & set(protocol.get("development_exposed_sequences", [])), "Known exposed sequences cannot enter validation/test")
        sequences = manifest.get("sequences", {})
        check("complete inventory", set(sequences) == set.union(*sets) and sum(len(v.get("frames", [])) for v in sequences.values()) == 33600, "Expected 112 sequences and 33,600 inventoried frames")
        seen = {}; overlaps = 0; inconsistent = 0
        for partition, ids in partitions.items():
            for sid in ids:
                record = sequences.get(sid, {})
                inconsistent += record.get("split") != partition
                for frame in record.get("frames", []):
                    fingerprint = frame.get("image_sha256")
                    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                        inconsistent += 1
                    if fingerprint in seen and seen[fingerprint] != partition:
                        overlaps += 1
                    seen[fingerprint] = partition
        check("inventory consistency", inconsistent == 0, f"Invalid entries: {inconsistent}")
        check("inventoried image overlap", overlaps == 0, f"Cross-partition hash collisions: {overlaps}; raw images are not rehashed by this audit")
    if source:
        for relative, fingerprint in source.items():
            match(f"current source: {relative}", contained(root, relative), fingerprint)
            match(f"source snapshot: {relative}", contained(study / "source", relative), fingerprint)

    complete = read("EVALUATION_COMPLETE.json")
    seal = decision = None
    if complete:
        seal = read("CHECKPOINTS_SEALED.json")
        decision = read("evaluation/gate-decision.json")
        if seal and protocol:
            expected_keys = {f"{seed}/{kind}" for seed in protocol.get("seeds", []) for kind in ("strfe", "graph")}
            check("all six checkpoints sealed", len(expected_keys) == 6 and set(seal.get("checkpoints", {})) == expected_keys, "Three seeds, two stages")
            for key, fingerprint in seal.get("checkpoints", {}).items():
                seed, kind = key.split("/")
                match(f"checkpoint {key}", contained(study, f"seed-{seed}/{kind}/best.pt"), fingerprint)
                training = read(f"seed-{seed}/{kind}/complete.json")
                if training:
                    check(f"full training completed: {key}", training.get("best_checkpoint_sha256") == fingerprint and protocol.get("min_epochs", 12) <= training.get("epochs", 0) <= protocol.get(f"{kind}_epochs", 40), "Completion record must match sealed best weights and frozen epoch range")
                history_path = contained(study, f"seed-{seed}/{kind}/history.json")
                try:
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    history_ok = isinstance(history, list) and bool(training) and len(history) == training.get("epochs") and [row["epoch"] for row in history] == list(range(1, len(history) + 1))
                except (OSError, ValueError, KeyError, TypeError):
                    history_ok = False
                check(f"epoch history: {key}", history_ok, "Contiguous saved epoch history must match training completion")
            match("sealed protocol", study / "protocol.json", seal.get("protocol_sha256"))
            match("sealed splits", study / "split-manifest.json", seal.get("manifest_sha256"))
        match("report", study / "REPORT.md", complete.get("report_sha256"))
        artifacts = complete.get("artifacts", {})
        required = {"evaluation/gate-decision.json", "evaluation/scene-horizon-metrics.json", "evaluation/comparisons-and-ci.json", "evaluation/failure-cases.csv", "evaluation/strata.csv"}
        normalized = {name.replace("\\", "/") for name in artifacts}
        check("required evaluation artifacts", required <= normalized, "Metrics, intervals, failures, strata and decision must be hashed")
        if protocol and manifest:
            predictions = {f"evaluation/predictions-{seed}-{sid}.npz" for seed in protocol.get("seeds", []) for sid in manifest.get("partitions", {}).get("test", [])}
            check("all test prediction artifacts", len(predictions) == 66 and predictions <= normalized, "22 sequences times three seeds")
        for relative, fingerprint in artifacts.items():
            match(f"artifact: {relative}", contained(study, relative), fingerprint)
        if decision and seal:
            match("decision checkpoint seal", study / "CHECKPOINTS_SEALED.json", decision.get("test_checkpoint_seal"))
            graph_pass = decision.get("gates", {}).get("strfe_graph", {}).get("pass")
            check("consistent completion decision", isinstance(graph_pass, bool) and complete.get("gate_pass") == graph_pass and complete.get("decision") == decision.get("decision"), "Completion and gate records must agree")

    verified = bool(complete) and all(item["pass"] for item in checks)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study": str(study.resolve()),
        "artifact_completion_verified": verified,
        "model_gate": decision if verified else None,
        "submission_readiness": "NOT ESTABLISHED: artifact verification cannot establish novelty or external validity",
        "audit_scope": "Stored hashes, sequence inventory, sealed checkpoints, saved epoch counts and output completeness. Does not independently verify training execution, rehash raw data or recalculate statistics; does not certify location independence or pretraining provenance.",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/evidence-20260907")
    parser.add_argument("--out", type=Path, default=ROOT / "publication/generated")
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 unless completed study artifacts pass the audit")
    args = parser.parse_args()
    study = args.study.resolve(); out = args.out.resolve()
    if out.is_relative_to(study) or study.is_relative_to(out):
        parser.error("Output must be separate from the study directory")
    result = audit(study)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Publication evidence audit", "", f"Generated: {result['generated_utc']}", "", f"Completed study artifacts verified: **{result['artifact_completion_verified']}**", "", result["submission_readiness"], "", result["audit_scope"], "", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    for row in result["checks"]:
        detail = row["detail"].replace("|", "\\|")
        lines.append(f"| {row['check']} | {'PASS' if row['pass'] else 'PENDING / FAIL'} | {detail} |")
    if result["model_gate"]:
        lines.extend(["", "## Verified model decision", "", result["model_gate"]["decision"]])
    (out / "evidence-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("artifact_completion_verified", "submission_readiness")}, indent=2))
    if args.require_complete and not result["artifact_completion_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
