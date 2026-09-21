"""Sanity checks for access_control.py and its wiring into agent/nodes.py.
Runs free/offline — no API key needed, no server required.
"""

import json
import tempfile
from pathlib import Path

import access_control
from access_control import ALL_CATEGORIES, KeyScope, _load_scoped_keys, resolve_scope
from agent.nodes import plan_node, synthesize_node

print("=== KeyScope.filter_categories() ===")
admin = KeyScope(role="admin", categories=[ALL_CATEGORIES], can_ingest=True)
assert admin.filter_categories(["HR", "Finance"]) == ["HR", "Finance"]
print("PASS: admin scope (['*']) passes every category through unchanged")

hr_scope = KeyScope(role="hr-team", categories=["HR", "General"], can_ingest=False)
assert hr_scope.filter_categories(["HR", "Finance", "General"]) == ["HR", "General"]
assert hr_scope.filter_categories(["Finance", "Legal"]) == []
print("PASS: restricted scope filters down to only its allowed categories")

print("\n=== _load_scoped_keys() loads a real config file ===")
tmp_dir = Path(tempfile.mkdtemp())
config_path = tmp_dir / "api_keys.json"
config_path.write_text(
    json.dumps(
        {
            "keys": [
                {"key": "test-hr-key-123", "role": "hr-team", "categories": ["HR", "General"], "can_ingest": False},
                {"key": "test-readonly-all-456", "role": "readonly-all", "categories": ["*"], "can_ingest": False},
            ]
        }
    )
)

original_path = access_control.ACCESS_CONTROL_CONFIG_PATH
access_control.ACCESS_CONTROL_CONFIG_PATH = str(config_path)
try:
    loaded = _load_scoped_keys()
finally:
    access_control.ACCESS_CONTROL_CONFIG_PATH = original_path

assert loaded["test-hr-key-123"].role == "hr-team"
assert loaded["test-hr-key-123"].categories == ["HR", "General"]
assert loaded["test-hr-key-123"].can_ingest is False
assert loaded["test-readonly-all-456"].categories == [ALL_CATEGORIES]
print("PASS: config file loaded correctly into KeyScope objects")

print("\n=== _load_scoped_keys() with no config path configured ===")
access_control.ACCESS_CONTROL_CONFIG_PATH = None
assert _load_scoped_keys() == {}
access_control.ACCESS_CONTROL_CONFIG_PATH = original_path
print("PASS: unset path -> no additional keys, not an error")

print("\n=== resolve_scope() (monkeypatched _SCOPED_KEYS, no server needed) ===")
original_scoped_keys = access_control._SCOPED_KEYS
access_control._SCOPED_KEYS = {"test-hr-key-123": hr_scope}
try:
    assert resolve_scope("test-hr-key-123") == hr_scope
    assert resolve_scope("totally-unknown-key") is None
finally:
    access_control._SCOPED_KEYS = original_scoped_keys
print("PASS: resolve_scope() finds a known scoped key and rejects an unknown one")

print("\n=== plan_node: allowed_categories=None means no restriction ===")
# monkeypatch plan_categories so this stays free/offline (no real LLM call)
import agent.nodes as nodes_module

original_plan_categories = nodes_module.plan_categories
nodes_module.plan_categories = lambda query: (["HR", "Finance"], False)
try:
    result = plan_node({"query": "test", "allowed_categories": None})
    assert result == {"categories": ["HR", "Finance"], "access_restricted": False, "off_topic": False}
    print("PASS: no allowed_categories -> router's decision passes through unchanged")

    print("\n=== plan_node: allowed_categories narrows the router's decision ===")
    result = plan_node({"query": "test", "allowed_categories": ["HR", "General"]})
    assert result == {"categories": ["HR"], "access_restricted": True, "off_topic": False}
    print("PASS: Finance dropped (not in allowed_categories), access_restricted=True")

    print("\n=== plan_node: allowed_categories that already covers everything requested ===")
    result = plan_node({"query": "test", "allowed_categories": ["HR", "Finance", "General"]})
    assert result == {"categories": ["HR", "Finance"], "access_restricted": False, "off_topic": False}
    print("PASS: nothing actually filtered out -> access_restricted=False")
finally:
    nodes_module.plan_categories = original_plan_categories

print("\n=== synthesize_node: distinguishes access-denied from genuinely-empty ===")
result = synthesize_node({"retrieved_chunks": [], "access_restricted": True, "off_topic": False, "query": "x"})
assert "doesn't have access" in result["answer"]
print(f"PASS: access_restricted=True -> {result['answer']!r}")

result = synthesize_node({"retrieved_chunks": [], "access_restricted": False, "off_topic": False, "query": "x"})
assert "couldn't find anything" in result["answer"]
print(f"PASS: access_restricted=False -> {result['answer']!r}")

result = synthesize_node({"retrieved_chunks": [], "access_restricted": False, "off_topic": True, "query": "x"})
assert "doesn't appear to relate" in result["answer"]
print(f"PASS: off_topic=True -> {result['answer']!r}")

import shutil

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nALL ACCESS CONTROL TESTS PASSED")
