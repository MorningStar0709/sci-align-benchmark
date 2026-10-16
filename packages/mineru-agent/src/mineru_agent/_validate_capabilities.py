import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]  # restored Phase 5
CAPABILITIES = ROOT / "benchmark" / "agent" / "agent_capabilities.json"



def _configure_stdout():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parameters(intent):
    params = intent.get("parameters")
    if params is None:
        return {}, []
    return params.get("properties", {}) or {}, params.get("required", []) or []


def _validate_meta(meta: dict, intents: list) -> list:
    """Validate meta section: total_intents, privacy_policy."""
    errors = []
    declared_total = meta.get("total_intents")
    if isinstance(declared_total, int) and declared_total != len(intents):
        errors.append(f"meta.total_intents mismatch: {declared_total} != {len(intents)}")
    privacy_policy = meta.get("privacy_policy", {})
    if privacy_policy.get("principle") != "cloud_for_public_or_sanitized_local_for_private":
        errors.append("meta.privacy_policy.principle missing or inconsistent")
    if not privacy_policy.get("cloud_allowed"):
        errors.append("meta.privacy_policy.cloud_allowed must list cloud-safe task classes")
    if not privacy_policy.get("local_only"):
        errors.append("meta.privacy_policy.local_only must list local-only task classes")
    return errors


_SPECIAL_NO_PROVIDER_DEFAULTS = {
    "arxiv_search",
    "corpus_parse",
    "leaderboard_viz",
    "mineru_parse",
    "multi_format_parse",
    "reproduce_all",
}
_SPECIAL_PROVIDER_REQUIREMENT = {"medical_rag"}


def _validate_intent(intent: dict, seen_ids: set) -> list:
    """Validate a single intent entry."""
    errors = []
    pid = intent.get("intent_id")
    if not pid:
        errors.append("intent missing intent_id")
        return errors
    if pid in seen_ids:
        errors.append(f"{pid}: duplicate intent_id")
    seen_ids.add(pid)

    params = intent.get("parameters")
    if params is not None:
        if params.get("additionalProperties") is not False:
            errors.append(f"{pid}: missing or incorrect additionalProperties: false")
        if not isinstance(params.get("properties"), dict):
            errors.append(f"{pid}: parameters.properties must be an object")

    if pid in _SPECIAL_NO_PROVIDER_DEFAULTS:
        if "provider_defaults" in intent:
            errors.append(f"{pid}: should NOT have provider_defaults (no-API/local-parser intent)")
    elif pid in _SPECIAL_PROVIDER_REQUIREMENT:
        if "provider_requirements" not in intent:
            errors.append(f"{pid}: missing provider_requirements")
    elif "provider_defaults" not in intent:
        errors.append(f"{pid}: missing provider_defaults")

    output = intent.get("output", {})
    if "artifacts" not in output:
        errors.append(f"{pid}: output missing artifacts")

    return errors


def _validate_intents(d: dict) -> list:
    """Validate all intents: IDs, parameters, provider config, output."""
    errors = []
    seen_ids = set()
    for intent in d.get("intents", []):
        errors.extend(_validate_intent(intent, seen_ids))
    return errors


def _validate_resource_mappings(intents: list) -> list:
    """Cross-validate intent IDs against resource_scheduler mappings."""
    errors = []
    try:
        from mineru_agent.resource_scheduler import AUTO_PROVIDER_MAPPING, LOCAL_PROVIDER_MAPPING
        intent_ids = {intent.get("intent_id") for intent in intents if intent.get("intent_id")}
        auto_ids = set(AUTO_PROVIDER_MAPPING)
        local_ids = set(LOCAL_PROVIDER_MAPPING)
        if auto_ids != intent_ids:
            errors.append(f"AUTO_PROVIDER_MAPPING mismatch: missing={sorted(intent_ids - auto_ids)} extra={sorted(auto_ids - intent_ids)}")
        if local_ids != intent_ids:
            errors.append(f"LOCAL_PROVIDER_MAPPING mismatch: missing={sorted(intent_ids - local_ids)} extra={sorted(local_ids - intent_ids)}")
    except Exception as exc:
        errors.append(f"resource_scheduler mapping validation failed: {exc}")
    return errors


def _validate_scheduler(scheduler: dict) -> list:
    """Validate scheduler config: fallback_order, provider_parameter_map."""
    errors = []
    fallback_order = scheduler.get("fallback_order", {})
    for resource_type in ["api", "local_api", "local_gpu", "script"]:
        if resource_type not in fallback_order:
            errors.append(f"fallback_order missing resource_type: {resource_type}")

    provider_map = scheduler.get("provider_parameter_map", {})
    for provider_name, provider_config in provider_map.items():
        for subname, subconfig in provider_config.items():
            if "model" not in subconfig:
                errors.append(f"provider {provider_name}.{subname}: missing model")
            if "default_temperature" not in subconfig:
                errors.append(f"provider {provider_name}.{subname}: missing default_temperature")
            if "default_max_tokens" not in subconfig:
                errors.append(f"provider {provider_name}.{subname}: missing default_max_tokens")
    return errors


def _print_intents(intents: list):
    """Print a summary of intents."""
    for intent in intents:
        props, required = _parameters(intent)
        print(
            f"  {intent.get('rank', -1):2d}. {intent.get('intent_id', ''):<25s} "
            f"resource={intent.get('resource_type', ''):<10s} "
            f"params={len(props):2d} required={required}"
        )


def main():
    """Validate capabilities JSON. Split into focused helpers for lower CC."""
    _configure_stdout()
    d = _load_json(CAPABILITIES)

    meta = d.get("meta", {})
    intents = d.get("intents", [])
    print("[OK] JSON valid")
    print(f"   Project: {meta.get('project', '')}")
    print(f"   Schema: {meta.get('schema_version', '')}, Intent: {meta.get('intent_version', '')}")
    print(f"   Intents: {meta.get('total_intents', len(intents))}")
    print()

    _print_intents(intents)

    print()
    scheduler = d.get("resource_scheduler_config", {})
    print(f"   Resource types: {list((scheduler.get('resource_types') or {}).keys())}")
    print(f"   Providers:      {list((scheduler.get('provider_parameter_map') or {}).keys())}")

    errors = []
    errors.extend(_validate_meta(meta, intents))
    errors.extend(_validate_intents(d))
    errors.extend(_validate_resource_mappings(intents))
    errors.extend(_validate_scheduler(scheduler))

    if errors:
        print("\n[FAIL] ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print(f"\n   Total: {len(errors)} errors")
        return 1

    print("\n[OK] All deep checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
