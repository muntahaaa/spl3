"""
verify_pipeline.py
==================
End-to-end verification script for the triplet reasoning pipeline.

Run this AFTER you have stored the JSON state to Neo4j (Tab ③) and BEFORE
or AFTER running chain_understand (Tab ④) to confirm every stage is healthy.

Usage:
    python verify_pipeline.py --state log/json_state/state_20260425_194816.json

What it checks (in order):
    STAGE 1 — DB Connectivity
    STAGE 2 — Expected nodes exist in Neo4j (Pages + Elements)
    STAGE 3 — Expected relationships exist (HAS_ELEMENT, LEADS_TO)
    STAGE 4 — get_chain_start_nodes() returns the correct start page
    STAGE 5 — get_chain_from_start() returns the correct number of triplets
    STAGE 6 — Every triplet has a valid element dict (never None)
    STAGE 7 — Every triplet has a valid action_name (never empty)
    STAGE 8 — Dry-run triplet reasoning on the FIRST triplet only (calls the LLM)
    STAGE 9 — After chain_understand has run: verify enriched descriptions in DB
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── Allow running from the project root ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from data.graph_db import Neo4jDatabase
from chain_understand import (
    create_triplet_reasoning_chain,
    _resolve_element,
    _resolve_action_name,
    process_triplet,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
INFO = "   ℹ"


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def result(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    line = f"  {icon}  {label}"
    if detail:
        line += f"\n{INFO}  {detail}"
    print(line)
    return ok


def warn(label: str, detail: str = ""):
    line = f"  {WARN}  {label}"
    if detail:
        line += f"\n{INFO}  {detail}"
    print(line)


def info(msg: str):
    print(f"  {INFO}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 0  Parse the state JSON and derive what SHOULD be in Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def parse_state(state_path: str) -> Dict[str, Any]:
    """
    Parse the state JSON and derive:
      - expected_pages:    list of source_page filenames from history_steps
                           + the final_page screenshot
      - expected_steps:    list of {step, action, element_number, source_page}
      - task:              task description string

    From the Weather state JSON we expect:
      Step 0: swipe  (element 16) — Weather_step1 screenshot (pre-action)
      Step 1: tap    (element 10) — Weather_step1 screenshot (post-swipe)
      Step 2: text   (element  9) — Weather_step2
      Step 3: tap    (element 55) — Weather_step3
      + final page: Weather_step4
    """
    with open(state_path, "r", encoding="utf-8") as fh:
        state = json.load(fh)

    task       = state.get("tsk", "")
    app_name   = state.get("app_name", "")
    steps      = state.get("history_steps", [])
    final_page = state.get("final_page", {})

    parsed_steps = []
    for s in steps:
        tool_result = s.get("tool_result", {})
        parsed_steps.append({
            "step":           s["step"],
            "action":         tool_result.get("action", "unknown"),
            "element_number": s.get("recommended_action", ""),
            "source_page":    s.get("source_page", ""),
            "source_json":    s.get("source_json", ""),
            "timestamp":      s.get("timestamp", ""),
            "clicked_x":      tool_result.get("clicked_element", {}).get("x"),
            "clicked_y":      tool_result.get("clicked_element", {}).get("y"),
        })

    return {
        "task":         task,
        "app_name":     app_name,
        "steps":        parsed_steps,
        "final_page":   final_page,
        "num_steps":    len(steps),
        # We expect num_steps triplets (one per hop)
        "expected_triplets": len(steps),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1  DB Connectivity
# ─────────────────────────────────────────────────────────────────────────────

def check_connectivity(db: Neo4jDatabase) -> bool:
    section("STAGE 1 — Database Connectivity")
    try:
        db.driver.verify_connectivity()
        return result(
            "Neo4j connection",
            True,
            f"URI={config.Neo4j_URI} DB={db.database}",
        )
    except Exception as e:
        result("Neo4j connection", False, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2  Node existence
# ─────────────────────────────────────────────────────────────────────────────

def check_nodes(db: Neo4jDatabase, parsed: Dict[str, Any]) -> bool:
    section("STAGE 2 — Node Existence in Neo4j")

    all_ok = True

    # Count all Page and Element nodes in DB
    with db.driver.session(database=db.database) as session:
        page_count = session.run("MATCH (p:Page) RETURN count(p) as c").single()["c"]
        elem_count = session.run("MATCH (e:Element) RETURN count(e) as c").single()["c"]

    info(f"Total Page nodes in DB  : {page_count}")
    info(f"Total Element nodes in DB: {elem_count}")

    # We expect at least num_steps + 1 pages (one per source + final)
    expected_min_pages = parsed["num_steps"] + 1
    ok = page_count >= expected_min_pages
    all_ok &= result(
        f"At least {expected_min_pages} Page nodes",
        ok,
        f"Found {page_count} (one per step source + final page)",
    )

    # We expect at least num_steps elements (one interacted element per step)
    ok = elem_count >= parsed["num_steps"]
    all_ok &= result(
        f"At least {parsed['num_steps']} Element nodes",
        ok,
        f"Found {elem_count}",
    )

    # Print all Page node IDs for reference
    with db.driver.session(database=db.database) as session:
        pages = session.run(
            "MATCH (p:Page) RETURN p.page_id as pid, p.description as desc "
            "ORDER BY p.timestamp LIMIT 20"
        )
        info("Page IDs in DB (up to 20, ordered by timestamp):")
        for rec in pages:
            desc = (rec["desc"] or "")[:60]
            print(f"       {rec['pid']}  →  {desc}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3  Relationship existence
# ─────────────────────────────────────────────────────────────────────────────

def check_relationships(db: Neo4jDatabase, parsed: Dict[str, Any]) -> bool:
    section("STAGE 3 — Relationships in Neo4j")

    all_ok = True

    with db.driver.session(database=db.database) as session:
        has_elem = session.run("MATCH ()-[r:HAS_ELEMENT]->() RETURN count(r) as c").single()["c"]
        leads_to = session.run("MATCH ()-[r:LEADS_TO]->() RETURN count(r) as c").single()["c"]

    info(f"HAS_ELEMENT  relationships: {has_elem}")
    info(f"LEADS_TO     relationships: {leads_to}")

    # Count action types from parsed steps
    action_counts: Dict[str, int] = {}
    for s in parsed["steps"]:
        action_counts[s["action"]] = action_counts.get(s["action"], 0) + 1

    info(f"Action breakdown from state JSON: {action_counts}")

    element_actions = sum(v for k, v in action_counts.items() if k in ("tap", "text", "long_press", "swipe", "back"))

    ok = leads_to >= element_actions
    all_ok &= result(
        f"LEADS_TO ≥ {element_actions} (tap/text/long_press steps)",
        ok,
        f"Found {leads_to}",
    )

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4  get_chain_start_nodes
# ─────────────────────────────────────────────────────────────────────────────

def check_start_nodes(db: Neo4jDatabase) -> List[Dict[str, Any]]:
    section("STAGE 4 — Chain Start Nodes")

    start_nodes = db.get_chain_start_nodes()
    ok = len(start_nodes) > 0
    result(
        f"get_chain_start_nodes() returned {len(start_nodes)} node(s)",
        ok,
        "Should be at least 1. If 0, the first page was not stored or "
        "it has an unexpected incoming LEADS_TO edge.",
    )

    for n in start_nodes:
        info(f"page_id={n.get('page_id')}  desc={str(n.get('description',''))[:60]}")

    if not start_nodes:
        print(f"\n  {FAIL}  Cannot continue — no start nodes found.")
        print(f"  {INFO}  Run Tab ③ first to store the JSON state to Neo4j.")

    return start_nodes


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5  get_chain_from_start — count and shape
# ─────────────────────────────────────────────────────────────────────────────

def check_chain(
    db: Neo4jDatabase,
    start_page_id: str,
    expected_triplets: int,
) -> List[Dict[str, Any]]:
    section("STAGE 5 — Chain Retrieval")

    chain = db.get_chain_from_start(start_page_id)
    ok = len(chain) == expected_triplets
    result(
        f"Chain has {len(chain)} triplet(s) (expected {expected_triplets})",
        ok,
        f"start_page_id={start_page_id}",
    )

    if len(chain) != expected_triplets:
        if len(chain) < expected_triplets:
            warn(
                "Fewer triplets than expected",
                "Possible causes:\n"
                "       • Some LEADS_TO edges are missing\n"
                "       • Re-run Tab ③ to re-store the session",
            )
        else:
            warn(
                "More triplets than expected",
                "Possible cause: duplicate pages stored for the same step. "
                "Check for duplicate page_id values in the DB.",
            )

    for i, hop in enumerate(chain):
        src  = hop["source_page"].get("page_id", "?")
        tgt  = hop["target_page"].get("page_id", "?")
        htype = hop.get("hop_type", "?")
        aname = _resolve_action_name(hop)
        elem  = _resolve_element(hop)
        eid   = elem.get("element_id", "(empty)")
        info(f"Triplet {i}: [{htype}] src={src}  action={aname}  elem={eid}  tgt={tgt}")

    return chain


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6  Element dict validity
# ─────────────────────────────────────────────────────────────────────────────

def check_element_dicts(chain: List[Dict[str, Any]]) -> bool:
    section("STAGE 6 — Element Dict Validity (never None)")

    all_ok = True
    for i, hop in enumerate(chain):
        elem = _resolve_element(hop)

        # After our fix, element should always be a dict
        ok_type = isinstance(elem, dict)
        all_ok &= result(
            f"Triplet {i}: element is a dict",
            ok_type,
            f"hop_type={hop.get('hop_type')}  raw={type(hop.get('element')).__name__}",
        )
        if not ok_type:
            continue

        eid = elem.get("element_id", "")
        if eid:
            result(f"Triplet {i}: element_id populated", True, eid)
        else:
            warn(
                f"Triplet {i}: element_id is empty",
                "Element resolution failed for this hop; re-store the session (Tab ③).",
            )

        # element must have a description field
        desc = elem.get("description", "")
        ok = bool(desc)
        all_ok &= result(
            f"Triplet {i}: element has description",
            ok,
            desc[:80] if desc else "(empty — element was stored without description)",
        )

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 7  Action name validity
# ─────────────────────────────────────────────────────────────────────────────

def check_action_names(chain: List[Dict[str, Any]]) -> bool:
    section("STAGE 7 — Action Name Validity (never empty)")

    all_ok = True
    for i, hop in enumerate(chain):
        aname = _resolve_action_name(hop)
        ok = bool(aname)
        all_ok &= result(
            f"Triplet {i}: action_name = '{aname}'",
            ok,
            f"raw action dict keys: {list(hop.get('action', {}).keys())}",
        )
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 8  Dry-run LLM reasoning on first triplet
# ─────────────────────────────────────────────────────────────────────────────

async def check_llm_reasoning(chain: List[Dict[str, Any]]) -> bool:
    section("STAGE 8 — Dry-Run LLM Reasoning (first triplet only)")

    if not chain:
        result("LLM reasoning dry-run", False, "Chain is empty — nothing to test")
        return False

    triplet = chain[0]
    elem   = _resolve_element(triplet)
    aname  = _resolve_action_name(triplet)

    info(f"Testing triplet 0:")
    info(f"  source_page : {triplet['source_page'].get('page_id', '?')}")
    info(f"  element     : {elem.get('element_id', '(empty)')}  →  {elem.get('description','')[:60]}")
    info(f"  action_name : {aname}")
    info(f"  target_page : {triplet['target_page'].get('page_id', '?')}")

    try:
        reasoning_chain = create_triplet_reasoning_chain()
        processed = await process_triplet(triplet, reasoning_chain)

        if "reasoning_error" in processed:
            result("LLM reasoning returned result", False, processed["reasoning_error"])
            return False

        reasoning = processed.get("reasoning", {})
        required_keys = [
            "context", "user_intent", "state_change", "task_relation",
            "source_page_enhanced_desc", "element_enhanced_desc", "target_page_enhanced_desc",
        ]
        missing = [k for k in required_keys if not reasoning.get(k)]

        if missing:
            result(
                "LLM reasoning output has all required fields",
                False,
                f"Missing or empty: {missing}",
            )
            return False

        result("LLM reasoning returned valid JSON with all 7 fields", True)

        info("Reasoning output preview:")
        info(f"  context     : {reasoning['context'][:80]}")
        info(f"  user_intent : {reasoning['user_intent'][:80]}")
        info(f"  state_change: {reasoning['state_change'][:80]}")
        info(f"  element_desc: {reasoning['element_enhanced_desc'][:80]}")
        info(f"  src_page_desc (new): {reasoning['source_page_enhanced_desc'][:80]}")
        info(f"  tgt_page_desc (new): {reasoning['target_page_enhanced_desc'][:80]}")

        return True

    except Exception as e:
        result("LLM reasoning dry-run", False, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 9  Post-chain_understand: verify enriched descriptions in DB
# ─────────────────────────────────────────────────────────────────────────────

def check_post_understand(db: Neo4jDatabase, chain: List[Dict[str, Any]]) -> bool:
    section("STAGE 9 — Post chain_understand: Enriched Descriptions in DB")
    info("(Only meaningful AFTER chain_understand has run successfully)")

    all_ok = True
    with db.driver.session(database=db.database) as session:
        for i, hop in enumerate(chain):
            elem = _resolve_element(hop)
            eid  = elem.get("element_id", "")

            # Check page description was enriched
            src_id = hop["source_page"].get("page_id", "")
            rec = session.run(
                "MATCH (p:Page {page_id: $pid}) RETURN p.description as d",
                pid=src_id,
            ).single()
            desc = (rec["d"] or "") if rec else ""
            ok = len(desc) > 30  # enriched descriptions are substantive
            if ok:
                result(f"Triplet {i}: source_page description enriched", True, desc[:80])
            else:
                warn(
                    f"Triplet {i}: source_page description short/empty",
                    f"'{desc}' — chain_understand may not have run yet for this triplet",
                )

            # Check element reasoning was saved
            if eid:
                rec2 = session.run(
                    "MATCH (e:Element {element_id: $eid}) RETURN e.reasoning as r",
                    eid=eid,
                ).single()
                reasoning_raw = (rec2["r"] or "") if rec2 else ""
                ok2 = bool(reasoning_raw)
                if ok2:
                    result(f"Triplet {i}: element reasoning saved to DB", True, eid)
                else:
                    warn(
                        f"Triplet {i}: element reasoning not yet in DB",
                        f"element_id={eid} — run chain_understand first",
                    )
                all_ok &= ok2
            else:
                warn(f"Triplet {i}: skipping reasoning check — no element_id")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(state_path: str, skip_llm: bool = False):
    print("\n" + "═" * 60)
    print("  TRIPLET REASONING PIPELINE — VERIFICATION REPORT")
    print("═" * 60)

    # ── Parse state JSON ──────────────────────────────────────────────────────
    if not os.path.exists(state_path):
        print(f"\n{FAIL}  State file not found: {state_path}")
        sys.exit(1)

    parsed = parse_state(state_path)
    print(f"\n  Task     : {parsed['task']}")
    print(f"  App      : {parsed['app_name']}")
    print(f"  Steps    : {parsed['num_steps']}")
    print(f"  Expected triplets : {parsed['expected_triplets']}")
    print(f"\n  Step breakdown:")
    for s in parsed["steps"]:
        print(f"    step {s['step']}: {s['action']}  src={s['source_page'].split('/')[-1]}")
    print(f"    final page: {parsed['final_page'].get('screenshot','').split(chr(92))[-1]}")

    # ── Connect to DB ─────────────────────────────────────────────────────────
    db = Neo4jDatabase(
        config.Neo4j_URI,
        config.Neo4j_AUTH,
        database=config.Neo4j_DB,
    )

    scores: Dict[str, bool] = {}

    scores["connectivity"] = check_connectivity(db)
    if not scores["connectivity"]:
        print(f"\n{FAIL}  Cannot connect to Neo4j — aborting.")
        db.close()
        return

    scores["nodes"]         = check_nodes(db, parsed)
    scores["relationships"] = check_relationships(db, parsed)

    start_nodes = check_start_nodes(db)
    scores["start_nodes"] = bool(start_nodes)
    if not start_nodes:
        db.close()
        return

    # Use the first start node (or override via --page-id flag)
    start_page_id = start_nodes[0]["page_id"]
    info(f"Using start_page_id: {start_page_id}")

    chain = check_chain(db, start_page_id, parsed["expected_triplets"])
    scores["chain"] = len(chain) > 0

    if chain:
        scores["element_dicts"] = check_element_dicts(chain)
        scores["action_names"]  = check_action_names(chain)

        if not skip_llm:
            scores["llm_reasoning"] = await check_llm_reasoning(chain)
        else:
            warn("LLM reasoning dry-run skipped (--no-llm flag)")

        scores["post_understand"] = check_post_understand(db, chain)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY")
    all_pass = True
    for stage, ok in scores.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {stage}")
        all_pass &= ok

    print(f"\n{'═' * 60}")
    if all_pass:
        print("  ✅  ALL CHECKS PASSED — pipeline is healthy")
    else:
        print("  ❌  SOME CHECKS FAILED — see details above")
    print("═" * 60 + "\n")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the triplet reasoning pipeline against a stored state JSON"
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Path to the JSON state file (e.g. log/json_state/state_20260425_194816.json)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM dry-run (Stage 8) to avoid API costs during quick checks",
    )
    args = parser.parse_args()
    asyncio.run(main(args.state, skip_llm=args.no_llm))