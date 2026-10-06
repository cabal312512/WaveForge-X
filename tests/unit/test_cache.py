"""Scientific cache invalidation includes dirty source, models, and output integrity."""

import json

import pytest

from waveforge6g.config_dynamic import parse_dynamic_config
from waveforge6g.experiments import research_cache


def _complete_run(root, key):
    directory = root / "results" / "test_run"
    directory.mkdir(parents=True)
    for name in ("config.yaml", "trajectory.csv", "oracle_outcomes.csv", "summary.json"):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "metadata.json").write_text(
        json.dumps({"cache_key": key, "status": "complete"}), encoding="utf-8"
    )
    return directory


def test_implementation_digest_changes_for_edit_add_remove_and_rename(tmp_path):
    first_file = tmp_path / "one.py"
    first_file.write_text("x = 1", encoding="utf-8")
    first = research_cache.implementation_hash(tmp_path)
    first_file.write_text("x = 2", encoding="utf-8")
    edited = research_cache.implementation_hash(tmp_path)
    assert edited != first
    second_file = tmp_path / "two.py"
    second_file.write_text("y = 3", encoding="utf-8")
    added = research_cache.implementation_hash(tmp_path)
    assert added != edited
    second_file.unlink()
    assert research_cache.implementation_hash(tmp_path) == edited
    first_file.rename(tmp_path / "renamed.py")
    assert research_cache.implementation_hash(tmp_path) != edited


def test_cache_key_covers_seed_config_source_revision_dependencies_and_model(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    metadata = {
        "git_commit": "abc",
        "dependencies": {"numpy": "2.0"},
        "python_version": "3.12",
        "platform": "test",
        "thread_environment": {"OMP_NUM_THREADS": "1"},
    }
    monkeypatch.setattr(research_cache, "collect_metadata", lambda *_: metadata)
    monkeypatch.setattr(research_cache, "implementation_hash", lambda: "source_v1")
    config = parse_dynamic_config({})
    first = research_cache.cache_key(config, 1)
    assert first == research_cache.cache_key(config, 1)
    assert first != research_cache.cache_key(config, 2)
    changed = parse_dynamic_config({"observation": {"snr_std": 2}})
    assert first != research_cache.cache_key(changed, 1)
    monkeypatch.setattr(research_cache, "implementation_hash", lambda: "source_v2")
    assert first != research_cache.cache_key(config, 1)
    second = research_cache.cache_key(config, 1)
    metadata["git_commit"] = "def"
    assert second != research_cache.cache_key(config, 1)
    third = research_cache.cache_key(config, 1)
    metadata["dependencies"]["numpy"] = "2.1"
    assert third != research_cache.cache_key(config, 1)
    model = tmp_path / "model.json"
    model.write_text('{"weights": [1]}', encoding="utf-8")
    offline = parse_dynamic_config(
        {"selectors": [{"name": "offline", "type": "offline", "model": "model.json"}]}
    )
    before = research_cache.cache_key(offline, 1)
    model.write_text('{"weights": [2]}', encoding="utf-8")
    assert before != research_cache.cache_key(offline, 1)


def test_cache_requires_complete_unchanged_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    key = "a" * 64
    directory = _complete_run(tmp_path, key)
    assert research_cache.lookup(key) is None
    research_cache.remember(key, directory)
    assert research_cache.lookup(key) == directory
    original = (directory / "trajectory.csv").read_text()
    (directory / "trajectory.csv").write_text("modified results", encoding="utf-8")
    assert research_cache.lookup(key) is None
    (directory / "trajectory.csv").write_text(original, encoding="utf-8")
    assert research_cache.lookup(key) == directory
    (directory / "summary.json").unlink()
    assert research_cache.lookup(key) is None


def test_cache_rejects_incomplete_runs_external_paths_and_malformed_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEFORGE_PROJECT_ROOT", str(tmp_path))
    key = "b" * 64
    directory = _complete_run(tmp_path, key)
    (directory / "metadata.json").write_text(
        json.dumps({"cache_key": key, "status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        research_cache.remember(key, directory)
    with pytest.raises(ValueError):
        research_cache.remember(key, tmp_path / "external")
    with pytest.raises(ValueError):
        research_cache.remember("../../arbitrary", directory)
    assert research_cache.lookup("../../arbitrary") is None
    pointer = tmp_path / ".cache" / "experiments" / f"{key}.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"directory": str(tmp_path / "external")}), encoding="utf-8")
    assert research_cache.lookup(key) is None
    pointer.write_text("[]", encoding="utf-8")
    assert research_cache.lookup(key) is None
