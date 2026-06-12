from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import config
from data.graph_db import Neo4jDatabase
from llm_rate_limit import wait_for_llm_slot
from nvidia_llm_bridge import NvidiaBridge

# ── LangSmith tracing ────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"]   = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = "ChainEvolve"

# ── Database ─────────────────────────────────────────────────────────────────
db = Neo4jDatabase(config.Neo4j_URI, config.Neo4j_AUTH, database=config.Neo4j_DB)

# ── NVIDIA NIM bridge (direct — no Firebase worker needed) ────────────────────
bridge = NvidiaBridge()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_action_name(triplet: Dict[str, Any]) -> str:
    action = triplet.get("action") or {}
    return action.get("action_name") or action.get("action_type") or "unknown"


def _resolve_element(triplet: Dict[str, Any]) -> Dict[str, Any]:
    element = triplet.get("element")
    if element is None:
        action = triplet.get("action") or {}
        element = {
            "element_id":   "",
            "element_type": action.get("action_type", "unknown"),
            "description":  (
                f"{action.get('action_type', 'unknown')} action "
                f"(no element node found)"
            ),
        }
    return element


# ─────────────────────────────────────────────────────────────────────────────
#  Chain data extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_task_description(chain: List[Dict[str, Any]]) -> str:
    if not chain:
        return "Unknown task"
    try:
        other_info = chain[0]["source_page"].get("other_info", {})
        if isinstance(other_info, str):
            other_info = json.loads(other_info)
        return other_info.get("task_info", {}).get("description", "Unknown task")
    except Exception as e:
        print(f"Error extracting task information: {e}")
        return "Unknown task"


def format_chain_operations(chain: List[Dict[str, Any]]) -> str:
    operations = []
    for i, triplet in enumerate(chain):
        source_page  = triplet["source_page"].get("description", "Unknown page")
        element      = _resolve_element(triplet)
        element_desc = element.get("description", "Unknown element")
        target_page  = triplet["target_page"].get("description", "Unknown page")
        action_name  = _resolve_action_name(triplet)
        operations.append(
            f"Step {i + 1}: On page 【{source_page}】, perform 【{action_name}】 "
            f"on 【{element_desc}】 to reach page 【{target_page}】."
        )
    return "\n".join(operations)


def extract_element_details(chain: List[Dict[str, Any]]) -> str:
    details = []
    for i, triplet in enumerate(chain):
        element     = _resolve_element(triplet)
        action_name = _resolve_action_name(triplet)
        details.append(
            f"Element {i + 1}:\n"
            f"  ID: {element.get('element_id', 'N/A')}\n"
            f"  Type: {element.get('element_type', 'Unknown type')}\n"
            f"  Description: {element.get('description', 'Unknown description')}\n"
            f"  Action: {action_name}"
        )
    return "\n".join(details)


def extract_reasoning_results(chain: List[Dict[str, Any]]) -> str:
    texts = []
    for i, triplet in enumerate(chain):
        if "reasoning" not in triplet:
            continue
        r = triplet["reasoning"]
        texts.append(
            f"Step {i + 1} reasoning:\n"
            f"  Context     : {r.get('context', 'N/A')}\n"
            f"  User intent : {r.get('user_intent', 'N/A')}\n"
            f"  State change: {r.get('state_change', 'N/A')}\n"
            f"  Task relation: {r.get('task_relation', 'N/A')}"
        )
    return "\n".join(texts) if texts else "No reasoning results available"


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_SYSTEM = (
    "You are an AI assistant specialized in evaluating whether UI operation chains "
    "can be templated into reusable high-level actions. The chain may include tap, "
    "text input, long press, swipe, and back navigation steps.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    "  is_templateable (bool), confidence_score (float 0-1), "
    "reason (str), suggested_name (str)\n"
    "No preamble, no markdown fences."
)


def _build_eval_user_prompt(task_description: str, chain_operations: str) -> str:
    return (
        f"Task description: {task_description}\n\n"
        f"Chain operations:\n{chain_operations}\n\n"
        "Evaluate from the following aspects:\n"
        "1. Does this chain have clear start and end steps?\n"
        "2. Does it have clear business logic and goals?\n"
        "3. Does it form a complete, meaningful task flow?\n"
        "4. Can it be reused in similar tasks?\n"
        "5. Are there obvious parameterisable parts?\n\n"
        "Return JSON only."
    )


_GEN_SYSTEM = (
    "You are an AI assistant specialized in generating high-level UI operation nodes. "
    "Generate a complete description of a high-level action node based on the chain "
    "information.  The chain may include tap, text input, long press, swipe, and back "
    "navigation — treat all as first-class actions.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    "  action_id (str, format: 'high_level_action_xxx'),\n"
    "  name (str),\n"
    "  description (str),\n"
    "  preconditions (list[str]),\n"
    "  element_sequence (list[dict] each with: element_id, order, atomic_action, action_params),\n"
    "  template_pattern (dict with: criteria, parameter_fields)\n"
    "No preamble, no markdown fences."
)


def _build_gen_user_prompt(
    task_description: str,
    chain_operations: str,
    element_details: str,
    reasoning_results: str,
) -> str:
    return (
        f"Task description: {task_description}\n\n"
        f"Chain operations:\n{chain_operations}\n\n"
        f"Chain element details:\n{element_details}\n\n"
        f"Chain reasoning results:\n{reasoning_results}\n\n"
        "Generate the high-level action node JSON now."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Core async processing functions
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_chain_templateability(
    chain: List[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Ask the model whether this chain can be templated.
    Returns (is_templateable, result_dict).
    """
    user_prompt = _build_eval_user_prompt(
        task_description=extract_task_description(chain),
        chain_operations=format_chain_operations(chain),
    )

    try:
        await wait_for_llm_slot()
        print("    [chain_evolve] Calling NVIDIA NIM for templateability eval...")
        result = await bridge.call_json(
            system_prompt=_EVAL_SYSTEM,
            user_prompt=user_prompt,
        )
        if "is_templateable" in result:
            return bool(result["is_templateable"]), result
        print(f"Warning: Unexpected evaluation result format: {result}")
        return False, None
    except Exception as e:
        print(f"Error evaluating the chain: {e}")
        return False, None


async def generate_action_node(
    chain: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Ask the model to generate a high-level action node description.
    Returns the parsed dict or None on failure.
    """
    user_prompt = _build_gen_user_prompt(
        task_description=extract_task_description(chain),
        chain_operations=format_chain_operations(chain),
        element_details=extract_element_details(chain),
        reasoning_results=extract_reasoning_results(chain),
    )

    try:
        await wait_for_llm_slot()
        print("    [chain_evolve] Calling NVIDIA NIM for action node generation...")
        result = await bridge.call_json(
            system_prompt=_GEN_SYSTEM,
            user_prompt=user_prompt,
        )
        if isinstance(result, dict) and "action_id" in result:
            return result
        print(f"Warning: Unexpected generation result format: {result}")
        return None
    except Exception as e:
        print(f"Error generating high-level action node: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Database write helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def create_action_node_in_db(action_data: Dict[str, Any]) -> Optional[str]:
    """Create a high-level action node in Neo4j."""
    try:
        properties = {
            "action_id":        action_data["action_id"],
            "name":             action_data["name"],
            "description":      action_data["description"],
            "preconditions":    json.dumps(action_data["preconditions"]),
            "element_sequence": json.dumps(action_data["element_sequence"]),
            "template_pattern": json.dumps(action_data["template_pattern"]),
            "is_high_level":    True,
        }
        node_id = db.create_action(properties)
        if not node_id:
            print("Failed to create high-level action node")
            return None
        print(f"Successfully created high-level action node, ID: {node_id}")
        return node_id
    except Exception as e:
        print(f"Error creating high-level action node: {e}")
        return None


def create_action_element_relations(action_data: Dict[str, Any]) -> bool:
    """
    Create COMPOSED_OF relationships between the high-level action and its
    constituent elements in Neo4j.
    """
    success = True
    for element_info in action_data.get("element_sequence", []):
        element_id = element_info.get("element_id", "")
        if not element_id:
            print(
                f"  Skipping COMPOSED_OF for step {element_info.get('order','?')} "
                f"— no element_id (legacy direct hop)"
            )
            continue

        ok = db.add_element_to_action(
            action_id=action_data["action_id"],
            element_id=element_id,
            order=element_info.get("order"),
            atomic_action=element_info.get("atomic_action"),
            action_params=element_info.get("action_params", {}),
        )
        if not ok:
            print(f"  Failed COMPOSED_OF for element_id={element_id}")
            success = False

    return success


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def evolve_chain_to_action(start_page_id: str) -> Optional[str]:
    """
    Process a full chain and evolve it into a high-level action node in Neo4j.

    Args:
        start_page_id: The page_id of the first page in the recorded chain.

    Returns:
        The action_id of the newly created high-level action node, or None.
    """
    try:
        # 1. Retrieve the chain
        print(f"Getting chain from start_page_id={start_page_id!r}...")
        chain = db.get_chain_from_start(start_page_id)
        if not chain:
            print(f"No chain found for start_page_id={start_page_id!r}")
            return None
        print(f"Retrieved {len(chain)} triplet(s)")

        # 2. Evaluate templateability
        print("Evaluating templateability...")
        is_templateable, evaluation_result = await evaluate_chain_templateability(chain)
        if not is_templateable:
            reason = (evaluation_result or {}).get("reason", "No reason provided")
            print(f"Chain is non-templatable: {reason}")
            return None
        print(
            f"Chain is templateable — confidence: "
            f"{evaluation_result.get('confidence_score', 0):.2f}, "
            f"suggested name: {evaluation_result.get('suggested_name', 'Unnamed')}"
        )

        # 3. Generate high-level action node content
        print("Generating high-level action node...")
        action_data = await generate_action_node(chain)
        if not action_data:
            print("Failed to generate action node content")
            return None
        print(f"Generated action node: {action_data['name']}")

        # 4. Persist action node
        print("Creating action node in Neo4j...")
        node_id = create_action_node_in_db(action_data)
        if not node_id:
            print("Failed to create action node in DB")
            return None

        # 5. Create element relationships
        print("Creating COMPOSED_OF relations...")
        relations_ok = create_action_element_relations(action_data)
        if not relations_ok:
            print("Some COMPOSED_OF relations could not be created")

        print(
            f"Chain evolution complete — action: {action_data['name']} "
            f"(ID: {action_data['action_id']})"
        )
        return action_data["action_id"]

    except Exception as e:
        print(f"Error in evolve_chain_to_action: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Test runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_test():
    print("\n===== Chain Evolution Test =====")
    start_nodes = db.get_chain_start_nodes()
    if not start_nodes:
        print("❌ No start nodes found — store a session to DB first (Tab ③)")
        return

    start_page_id = start_nodes[0]["page_id"]
    print(f"✓ Using start_page_id: {start_page_id}")

    action_id = await evolve_chain_to_action(start_page_id)
    if not action_id:
        print("\n❌ Chain evolution failed — check the logs above for details")
        return

    print(f"\n✓ Chain evolution succeeded! action_id={action_id}")
    print("\n===== Test Completed ✨ =====")


# if __name__ == "__main__":
#     asyncio.run(run_test())