import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import config
from data.graph_db import Neo4jDatabase
from llm_rate_limit import wait_for_llm_slot

for proxy_var in ("OPENAI_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(proxy_var, None)

# Configure environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "ChainEvolve"

# Initialize LLM model
model = ChatGroq(
    model=config.LLM_MODEL,
    google_api_key=config.LLM_API_KEY,
    max_retries=0,
)

# Initialize database connection
URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH, database=config.Neo4j_DB)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers  (mirror the same helpers in chain_understand for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_action_name(triplet: Dict[str, Any]) -> str:
    """
    Return a unified action name from the triplet's action dict.

    graph_db normalises both hop types so that action["action_name"] is always
    set.  This helper reads it with a fallback chain for legacy data.
    """
    action = triplet.get("action") or {}
    return (
        action.get("action_name")
        or action.get("action_type")
        or "unknown"
    )


def _resolve_element(triplet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the element dict from the triplet, building a minimal placeholder
    for any legacy hop that still carries element=None.
    """
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
#  Pydantic output models
# ─────────────────────────────────────────────────────────────────────────────

class ChainEvaluationResult(BaseModel):
    is_templateable: bool = Field(description="Whether the chain can be templated")
    confidence_score: float = Field(
        description="Confidence score for templateability (0-1)"
    )
    reason: str = Field(description="Reason and explanation for the evaluation")
    suggested_name: str = Field(description="Suggested name for the high-level action")


class ActionNodeGeneration(BaseModel):
    action_id: str = Field(description="High-level action node ID")
    name: str = Field(description="High-level action name")
    description: str = Field(description="Detailed description")
    preconditions: List[str] = Field(
        description="Preconditions for executing the high-level action"
    )
    element_sequence: List[Dict[str, Any]] = Field(
        description="Sequence of elements included in the high-level action"
    )
    template_pattern: Dict[str, Any] = Field(description="Template matching pattern")


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain chains
# ─────────────────────────────────────────────────────────────────────────────

def create_chain_evaluation_chain():
    """Create an LCEL chain for evaluating whether the chain can be templated."""
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant specialized in evaluating whether UI "
                    "operation chains can be templated. You need to analyze the given "
                    "UI operation chain — which may include tap, text input, long press, "
                    "swipe, and back navigation steps — and determine if it has the "
                    "potential for templating."
                ),
            ),
            (
                "human",
                (
                    "Please evaluate whether the following UI operation chain can be "
                    "templated into a high-level action:\n\n"
                    "Task description: {task_description}\n\n"
                    "Chain operations:\n{chain_operations}\n\n"
                    "Please evaluate from the following aspects:\n"
                    "1. Does this operation chain have clear start and end steps?\n"
                    "2. Do the operations in the chain have clear business logic and goals?\n"
                    "3. Do these operations form a complete and meaningful task flow?\n"
                    "4. Is it possible to reuse this chain in other similar tasks?\n"
                    "5. Are there obvious parameterizable parts?\n\n"
                    "Please return your evaluation results in a structured manner, "
                    "including the following fields:\n"
                    "- is_templateable: Whether it can be templated (boolean)\n"
                    "- confidence_score: Confidence score (float between 0-1)\n"
                    "- reason: Detailed evaluation reason\n"
                    "- suggested_name: If it can be templated, the suggested high-level "
                    "action name\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )

    parser = JsonOutputParser(pydantic_object=ChainEvaluationResult)
    prompt = evaluation_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    return RunnablePassthrough() | prompt | model | parser


def create_action_generation_chain():
    """Create an LCEL chain for generating high-level action node content."""
    generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant specialized in generating high-level UI "
                    "operation nodes. You need to generate a complete description of a "
                    "high-level action node based on the given chain information.  The "
                    "chain may include any mix of tap, text input, long press, swipe, "
                    "and back navigation steps — treat them all as first-class actions."
                ),
            ),
            (
                "human",
                (
                    "Please generate a high-level action node based on the following "
                    "UI operation chain information:\n\n"
                    "Task description: {task_description}\n\n"
                    "Chain operations:\n{chain_operations}\n\n"
                    "Chain element details:\n{element_details}\n\n"
                    "Chain reasoning results:\n{reasoning_results}\n\n"
                    "Please generate a complete description of the high-level action "
                    "node, including the following fields:\n"
                    "- action_id: Generate a unique ID (format: 'high_level_action_xxx')\n"
                    "- name: Concise name of the high-level action\n"
                    "- description: Detailed description of function, purpose, and "
                    "execution process\n"
                    "- preconditions: List of preconditions for executing this action\n"
                    "- element_sequence: Ordered list of elements, each containing:\n"
                    "  * element_id: Element ID\n"
                    "  * order: Order of operation\n"
                    "  * atomic_action: The action performed (tap/text/swipe/back/etc.)\n"
                    "  * action_params: Action parameters (if any)\n"
                    "- template_pattern: Template matching pattern with:\n"
                    "  * criteria: Applicable matching conditions\n"
                    "  * parameter_fields: Parameterizable fields and their descriptions\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )

    parser = JsonOutputParser(pydantic_object=ActionNodeGeneration)
    prompt = generation_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    return RunnablePassthrough() | prompt | model | parser


# ─────────────────────────────────────────────────────────────────────────────
#  Chain data extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_task_description(chain: List[Dict[str, Any]]) -> str:
    """Extract task description from the first hop's source page."""
    task_info = "Unknown task"
    if not chain:
        return task_info
    try:
        other_info = chain[0]["source_page"].get("other_info", {})
        if isinstance(other_info, str):
            other_info = json.loads(other_info)
        task_info = (
            other_info.get("task_info", {}).get("description", "Unknown task")
        )
    except Exception as e:
        print(f"Error extracting task information: {e}")
    return task_info


def format_chain_operations(chain: List[Dict[str, Any]]) -> str:
    """
    Format all hops in the chain as a numbered step description.

    Works for ALL action types — tap, text, long_press, swipe, back —
    because _resolve_element and _resolve_action_name handle every case.
    """
    operations = []
    for i, triplet in enumerate(chain):
        source_page = triplet["source_page"].get("description", "Unknown page")
        element     = _resolve_element(triplet)
        element_desc = element.get("description", "Unknown element")
        target_page = triplet["target_page"].get("description", "Unknown page")
        action_name = _resolve_action_name(triplet)

        operation = (
            f"Step {i + 1}: On page 【{source_page}】, perform 【{action_name}】 "
            f"on 【{element_desc}】 to reach page 【{target_page}】."
        )
        operations.append(operation)

    return "\n".join(operations)


def extract_element_details(chain: List[Dict[str, Any]]) -> str:
    """
    Extract detailed information for all elements in the chain.

    Handles every hop type uniformly — _resolve_element guarantees a valid
    dict for every triplet regardless of whether the hop is element_hop or
    direct_hop (swipe / back).
    """
    elements = []
    for i, triplet in enumerate(chain):
        element     = _resolve_element(triplet)
        action_name = _resolve_action_name(triplet)

        element_id   = element.get("element_id", "N/A")
        element_type = element.get("element_type", "Unknown type")
        element_desc = element.get("description", "Unknown description")

        detail = (
            f"Element {i + 1}:\n"
            f"  ID: {element_id}\n"
            f"  Type: {element_type}\n"
            f"  Description: {element_desc}\n"
            f"  Action: {action_name}"
        )
        elements.append(detail)

    return "\n".join(elements)


def extract_reasoning_results(chain: List[Dict[str, Any]]) -> str:
    """Extract LLM reasoning results that were written by chain_understand."""
    reasoning_texts = []
    for i, triplet in enumerate(chain):
        if "reasoning" not in triplet:
            continue
        reasoning = triplet["reasoning"]
        text = (
            f"Step {i + 1} reasoning:\n"
            f"  Context     : {reasoning.get('context', 'N/A')}\n"
            f"  User intent : {reasoning.get('user_intent', 'N/A')}\n"
            f"  State change: {reasoning.get('state_change', 'N/A')}\n"
            f"  Task relation: {reasoning.get('task_relation', 'N/A')}"
        )
        reasoning_texts.append(text)

    return "\n".join(reasoning_texts) if reasoning_texts else "No reasoning results available"


# ─────────────────────────────────────────────────────────────────────────────
#  Core async processing functions
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_chain_templateability(
    chain: List[Dict[str, Any]]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Evaluate whether the chain can be templated into a high-level action.

    Returns:
        (is_templateable, evaluation_result_dict)
    """
    evaluation_chain = create_chain_evaluation_chain()

    evaluation_input = {
        "task_description": extract_task_description(chain),
        "chain_operations": format_chain_operations(chain),
    }

    try:
        await wait_for_llm_slot()
        result = await evaluation_chain.ainvoke(evaluation_input)
        if isinstance(result, dict) and "is_templateable" in result:
            return result["is_templateable"], result
        print(f"Warning: Unexpected evaluation result format: {result}")
        return False, None
    except Exception as e:
        print(f"Error evaluating the chain: {e}")
        return False, None


async def generate_action_node(
    chain: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Generate high-level action node content from the processed chain.

    Returns:
        Action node dict or None on failure.
    """
    generation_chain = create_action_generation_chain()

    generation_input = {
        "task_description": extract_task_description(chain),
        "chain_operations": format_chain_operations(chain),
        "element_details":  extract_element_details(chain),
        "reasoning_results": extract_reasoning_results(chain),
    }

    try:
        await wait_for_llm_slot()
        result = await generation_chain.ainvoke(generation_input)
        if isinstance(result, dict) and "action_id" in result:
            return result
        print(f"Warning: Unexpected generation result format: {result}")
        return None
    except Exception as e:
        print(f"Error generating high-level action node: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Database write helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_action_node_in_db(action_data: Dict[str, Any]) -> Optional[str]:
    """Create a high-level action node in Neo4j."""
    try:
        properties = {
            "action_id":       action_data["action_id"],
            "name":            action_data["name"],
            "description":     action_data["description"],
            "preconditions":   json.dumps(action_data["preconditions"]),
            "element_sequence": action_data["element_sequence"],
            "template_pattern": json.dumps(action_data["template_pattern"]),
            "is_high_level":   True,
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

    Skips any element_sequence entry that carries an empty element_id — these
    correspond to legacy direct hops (swipe/back) whose element node was not
    stored in the DB.
    """
    success = True
    for element_info in action_data.get("element_sequence", []):
        element_id = element_info.get("element_id", "")
        if not element_id:
            print(
                f"  Skipping COMPOSED_OF for step {element_info.get('order', '?')} "
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

    Works for any chain regardless of the action types it contains (tap, text,
    long_press, swipe, back).  Every hop is treated uniformly as an element-
    based triplet: source_page → element → target_page.

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
    """Run a quick end-to-end test using the first available start node."""
    print("\n===== Chain Evolution Test =====")

    print("\n1. Getting start nodes...")
    start_nodes = db.get_chain_start_nodes()
    if not start_nodes:
        print("❌ No start nodes found — store a session to DB first (Tab ③)")
        return

    start_page_id = start_nodes[0]["page_id"]
    print(f"✓ Using start_page_id: {start_page_id}")

    print("\n2. Running chain evolution...")
    action_id = await evolve_chain_to_action(start_page_id)

    if not action_id:
        print("\n❌ Chain evolution failed — check the logs above for details")
        return

    print(f"\n✓ Chain evolution succeeded! action_id={action_id}")
    print("\n===== Test Completed ✨ =====")


#if __name__ == "__main__":
    #asyncio.run(run_test())