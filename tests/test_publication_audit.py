"""Negative checks for publication claims: missing and tampered evidence fails closed."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("publication_audit", Path(__file__).resolve().parents[1] / "scripts/audit-publication-evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_study_never_reports_success(tmp_path):
    result = module.audit(tmp_path)
    assert result["artifact_completion_verified"] is False
    assert result["model_gate"] is None


def test_completion_flag_alone_is_insufficient(tmp_path):
    (tmp_path / "EVALUATION_COMPLETE.json").write_text(json.dumps({"gate_pass": True, "decision": "Excellent results"}))
    result = module.audit(tmp_path)
    assert result["artifact_completion_verified"] is False
    assert result["model_gate"] is None


def test_tampered_snapshot_detected(tmp_path):
    (tmp_path / "source").mkdir()
    source = tmp_path / "model.py"
    source.write_text("original")
    (tmp_path / "source/model.py").write_text("changed")
    (tmp_path / "source-manifest.json").write_text(json.dumps({"model.py": module.digest(source)}))
    result = module.audit(tmp_path, root=tmp_path)
    row = next(r for r in result["checks"] if r["check"] == "source snapshot: model.py")
    assert not row["pass"]


def test_artifact_cannot_escape_study(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        module.contained(tmp_path, "../outside.pt")
