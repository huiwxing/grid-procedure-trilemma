"""Connectivity smoke check of the reader panel: one tiny call per pinned profile, made before any paid run.

Default: print the plan (endpoint, wire model, credential variable name, what each
call must confirm, per-call cost bound). Nothing is sent and no key is read beyond
checking whether the variable is set.

--execute sends one call per sendable profile (the 15 sendable profiles of the panel): the negative control (a
profile the client must refuse to configure) is only checked for that refusal, and a candidate (a profile without a
priced route) is listed and never sent. It is a PAID action: run it only once the spend is approved.
Call ledgers go to <out-dir>/<profile_id>.jsonl (--out-dir, default out/smoke; append-only; a rerun resumes).

--only <profile_id> (repeatable) restricts the plan and --execute to those profiles.
--ledger-tag <tag> writes <profile_id>__<tag>.jsonl instead, e.g. to rerun a profile
whose request changed: its old ledger refuses the new run identity.

Usage: py -3.12 scripts/smoke_plan.py [--json] [--only ID ...] [--execute --cap-usd 0.10 [--ledger-tag TAG]
       [--out-dir DIR]]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT
from procedures.readers import Cell, check_wire, estimate_cells, load_panel, load_price_book, run_cells, scenario
from procedures.readers.client import ConfigurationError, InferenceProfile
from procedures.readers.client.adapters import build_request
from procedures.readers.executor import upper_bound_usd

SMOKE_MESSAGES = ({"role": "user", "content": "Return only the following word, without explanation: READY"},)
SMOKE_MAX_OUTPUT_TOKENS = 512  # reasoning models may truncate; usage still returns
LEDGER_DIR = ROOT / "out" / "smoke"     # default directory of the call ledgers (--out-dir)
LEDGER_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
UNSENT_ROLES = ("negative", "candidate")
COMMON_CHECKS = ["usage fields returned (cache + reasoning keys)", "served_model / system_fingerprint values"]
PROVIDER_CHECKS = {
    "openai": ["max_completion_tokens accepted", "completion_tokens_details.reasoning_tokens present"],
    "anthropic": ["thinking type honoured (thinking blocks only when enabled/adaptive)",
                  "output_tokens_details.thinking_tokens reported"],
    "google-ai-studio": ["REST field generationConfig.thinkingConfig.thinkingLevel accepted",
                         "thoughtsTokenCount present", "paid-tier key"],
    "deepseek": ["thinking type honoured", "prompt_cache_hit/miss_tokens present", "served alias vs V4.1-Flash"],
    "huggingface": ["x-inference-provider equals pinned route", "billed unit price = catalog (HF billing page)"],
}


def smoke_cell() -> Cell:
    """The one smoke request (its item_id and stage are labels only; the request hash does not depend on them)."""
    return Cell(item_id="connectivity", stage="smoke/connectivity", messages=SMOKE_MESSAGES,
                generation={"max_output_tokens": SMOKE_MAX_OUTPUT_TOKENS})


def ledger_path(profile_id: str, tag: str | None = None, out_dir: Path | None = None) -> Path:
    return Path(out_dir or LEDGER_DIR) / (f"{profile_id}__{tag}.jsonl" if tag else f"{profile_id}.jsonl")


def build_plan(panel, book) -> list[dict]:
    rows = []
    for pp in panel.profiles.values():
        if pp.role == "negative":
            try:
                InferenceProfile.from_dict(pp.spec)
            except ConfigurationError as exc:
                refusal = str(exc)
            else:
                raise SystemExit(f"{pp.profile_id}: negative control was accepted")
            rows.append({"profile_id": pp.profile_id, "role": pp.role, "tier": pp.tier, "slot": pp.slot,
                         "action": "configuration refusal only (no request)", "refusal": refusal,
                         "bound_usd": 0.0, "expected_high_usd": 0.0, "cap_usd": 0.0, "checks": []})
            continue
        if pp.role == "candidate":
            rows.append({"profile_id": pp.profile_id, "role": pp.role, "tier": pp.tier, "slot": pp.slot,
                         "action": "not sent: candidate without a priced route", "bound_usd": 0.0,
                         "expected_high_usd": 0.0, "cap_usd": 0.0, "checks": []})
            continue
        client_profile = pp.client_profile()
        cell = smoke_cell()
        cell = replace(cell, messages=pp.messages_to_send(cell.messages))  # as run_cells will send it
        endpoint, payload = build_request(client_profile, list(cell.messages), cell.generation)
        check_wire(pp, endpoint, payload)
        bound = upper_bound_usd(cell, pp, book)
        expected = estimate_cells([cell], pp, book, scenario(pp, "high"))["usd"]
        extra = ["reasoning suppressed (reasoning_leak must be false)"] if not pp.reasoning_expected else []
        if pp.user_message_suffix is not None:
            extra.append(f"user_message_suffix {pp.user_message_suffix!r} on the last user message is honoured")
        rows.append({"profile_id": pp.profile_id, "role": pp.role, "tier": pp.tier, "slot": pp.slot,
                     "action": "send 1 request", "endpoint": endpoint, "wire_model": payload.get("model", pp.spec["model"]),
                     "user_message_suffix": pp.user_message_suffix,
                     "price_key": pp.price_key, "key_env": client_profile.api_key_env,
                     "key_set": bool(os.environ.get(client_profile.api_key_env or "")),
                     "max_output_tokens": SMOKE_MAX_OUTPUT_TOKENS, "bound_usd": bound, "expected_high_usd": expected,
                     "cap_usd": max(bound * client_profile.max_attempts, expected),   # every attempt reserved
                     "checks": COMMON_CHECKS + PROVIDER_CHECKS[pp.spec["provider"]] + extra})
    return rows


def print_plan(rows, panel, book) -> None:
    print(f"connectivity smoke plan - panel {panel.path.name} sha256 {panel.sha256[:12]}, "
          f"prices {book.path.name} sha256 {book.sha256[:12]}")
    print(f"{'profile':34} {'role':9} {'slot':4} {'key env (set?)':22} {'bound $':>9} {'cap $':>9}  wire model")
    for r in rows:
        if r["role"] in UNSENT_ROLES:
            what = f"refused: {r['refusal']}" if r["role"] == "negative" else r["action"]
            print(f"{r['profile_id']:34} {r['role']:9} {'-':4} {'-':22} {0:>9.5f} {0:>9.5f}  {what}")
            continue
        key = f"{r['key_env']} ({'yes' if r['key_set'] else 'no'})"
        print(f"{r['profile_id']:34} {r['role']:9} {r['slot'] or '-':4} {key:22} "
              f"{r['bound_usd']:>9.5f} {r['cap_usd']:>9.5f}  {r['wire_model']}")
    sendable = [r for r in rows if r["role"] not in UNSENT_ROLES]
    refused = sum(r["role"] == "negative" for r in rows)
    print(f"\n{len(rows)} profiles: {len(sendable)} calls + {refused} configuration-refusal check + "
          f"{len(rows) - len(sendable) - refused} unsent candidate; total cap ${sum(r['cap_usd'] for r in rows):.4f} "
          f"(per-attempt bound ${sum(r['bound_usd'] for r in rows):.4f})")
    print("Each call must confirm:")
    for r in sendable:
        print(f"  {r['profile_id']}: " + "; ".join(r["checks"]))
    print("\nNothing was sent. --execute is a paid action: run it only once the spend is approved.")


def execute(rows, panel, book, cap_usd: float, ledger_tag: str | None = None, out_dir: Path | None = None) -> int:
    total = sum(r["cap_usd"] for r in rows)
    if total > cap_usd:
        raise SystemExit(f"planned cap ${total:.4f} exceeds --cap-usd ${cap_usd:.4f}; nothing was sent")
    failures = 0
    for r in rows:
        if r["role"] in UNSENT_ROLES:
            continue
        pp = panel.get(r["profile_id"])
        if not r["key_set"]:
            # An unset credential is a configuration gap, not a provider result: record it,
            # keep the smoke incomplete (non-zero exit), and continue with the other profiles.
            print(json.dumps({"profile_id": pp.profile_id, "status": "skipped", "outcome": "credential_unset",
                              "key_env": r["key_env"]}, ensure_ascii=False))
            failures += 1
            continue
        report = run_cells([smoke_cell()], pp, cap_usd=r["cap_usd"], concurrency=1, book=book,
                           ledger_path=ledger_path(pp.profile_id, ledger_tag, out_dir))
        rec = report["records"][0]
        raw_keys = sorted((rec.get("usage") or {}).get("raw") or {})
        print(json.dumps({"profile_id": pp.profile_id, "status": rec.get("status"), "outcome": rec.get("outcome"),
                          "served_model": rec.get("served_model"), "system_fingerprint": rec.get("system_fingerprint"),
                          "observed_provider": rec.get("observed_provider"), "reasoning_leak": rec.get("reasoning_leak"),
                          "usage_raw_keys": raw_keys, "usd": rec.get("usd"), "text": rec.get("text")}, ensure_ascii=False))
        failures += rec.get("outcome") != "observed" or bool(rec.get("reasoning_leak"))
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument("--execute", action="store_true", help="PAID: send the calls (only once the spend is approved)")
    parser.add_argument("--cap-usd", type=float, default=0.10, help="total spend cap for --execute")
    parser.add_argument("--only", action="append", metavar="PROFILE_ID", help="restrict to this profile (repeatable)")
    parser.add_argument("--ledger-tag", help="write <profile_id>__<tag>.jsonl instead of <profile_id>.jsonl")
    parser.add_argument("--out-dir", type=Path, help="directory of the call ledgers (default out/smoke)")
    args = parser.parse_args(argv)
    if args.ledger_tag is not None and not LEDGER_TAG.fullmatch(args.ledger_tag):
        parser.error("--ledger-tag must be letters, digits, '_' or '-'")
    panel, book = load_panel(), load_price_book()
    if args.only:
        unknown = sorted(set(args.only) - set(panel.profiles))
        if unknown:
            parser.error(f"unknown profile ids {unknown}; choose from {sorted(panel.profiles)}")
    rows = [r for r in build_plan(panel, book) if not args.only or r["profile_id"] in args.only]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_plan(rows, panel, book)
    if args.execute:
        return execute(rows, panel, book, args.cap_usd, args.ledger_tag, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
