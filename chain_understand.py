from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
import json
import os
import base64
import traceback
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from data.graph_db import Neo4jDatabase
from llm_rate_limit import wait_for_llm_slot
import config

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "ChainEvolve"

assert config.LLM_API_KEY, "LLM_API_KEY is not set in config!"
model = ChatGroq(
    model=config.LLM_MODEL,
    google_api_key=config.LLM_API_KEY,
    max_retries=0,
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH
db = Neo4jDatabase(URI, AUTH, database=config.Neo4j_DB)

_LLM_ACCESS_DENIED = False


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_action_name(triplet: Dict[str, Any]) -> str:
    """
    Return a human-readable action name from the triplet's action dict.

    graph_db normalises both hop types so that action["action_name"] is always
    populated (LEADS_TO carries it natively; NAVIGATED_BY has it synthesised
    from action_type).  This helper is a single authoritative place to read it,
    with a fallback chain for any legacy data that predates the normalisation.
    """
    action = triplet.get("action") or {}
    return (
        action.get("action_name")
        or action.get("action_type")
        or ""
    )


def _resolve_element(triplet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the element dict from the triplet.

    After the graph_db update, element is always a dict (never None) for both
    element_hop and direct_hop.  This helper guards against legacy data that
    might still carry None and converts it to a minimal placeholder dict so
    that all downstream code can call .get() safely without any isinstance
    checks or special-casing.
    """
    element = triplet.get("element")
    if element is None:
        # Legacy row — build a minimal placeholder so .get() always works
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


def _load_image_to_base64(image_path: str) -> str:
    """Load an image file and return it as a data URL for multimodal prompts."""
    if not image_path:
        return "data:image/png;base64,"

    ext = os.path.splitext(str(image_path))[1].lower()
    mime = "image/png"
    if ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _is_access_denied_error(exc: Exception) -> bool:
    """Return True when the exception indicates provider/project access denial."""
    msg = str(exc).lower()
    return (
        "permissiondenied" in msg
        or (
            "403" in msg
            and (
                "denied access" in msg
                or "contact support" in msg
                or "permission denied" in msg
            )
        )
    )


def _mark_llm_access_denied(exc: Exception) -> None:
    """Set one-way fail-safe switch and print a concise one-time message."""
    global _LLM_ACCESS_DENIED
    if not _LLM_ACCESS_DENIED:
        _LLM_ACCESS_DENIED = True
        print(
            "LLM access denied (403). Skipping further LLM reasoning/merge calls "
            "for this run and continuing with existing descriptions."
        )
        print(f"Original error: {exc}")


def _print_exception_details(prefix: str, exc: Exception) -> None:
    """Print rich exception diagnostics to help identify provider access failures."""
    print(f"{prefix} | exception_type={type(exc).__name__}")
    print(f"{prefix} | exception_repr={repr(exc)}")
    print(f"{prefix} | exception_args={getattr(exc, 'args', None)}")
    print(f"{prefix} | cause={repr(getattr(exc, '__cause__', None))}")
    print(f"{prefix} | context={repr(getattr(exc, '__context__', None))}")

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        print(f"{prefix} | http_status={status_code}")

        body_printed = False
        try:
            body_json = response.json()
            print(f"{prefix} | response_json={json.dumps(body_json, ensure_ascii=True)}")
            body_printed = True
        except Exception:
            pass

        if not body_printed:
            try:
                print(f"{prefix} | response_text={getattr(response, 'text', None)}")
            except Exception:
                print(f"{prefix} | response_text=<unavailable>")

    print(f"{prefix} | traceback_start")
    traceback.print_exc()
    print(f"{prefix} | traceback_end")


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic model for reasoning output
# ─────────────────────────────────────────────────────────────────────────────

class TripletReasoning(BaseModel):
    context: str = Field(description="Context description of the operation")
    user_intent: str = Field(description="User intention analysis")
    state_change: str = Field(description="State change description")
    task_relation: str = Field(description="Relationship with the task")
    source_page_enhanced_desc: str = Field(
        description="Enhanced description of the source page"
    )
    element_enhanced_desc: str = Field(
        description="Enhanced description of the element"
    )
    target_page_enhanced_desc: str = Field(
        description="Enhanced description of the target page"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LangChain chains
# ─────────────────────────────────────────────────────────────────────────────

def create_triplet_reasoning_chain():
    """
    Create LCEL chain for triplet reasoning.

    The prompt is action-type-aware: it receives the action_name (which for
    swipe/back will be e.g. "swipe" or "back") and element_desc (which for
    those actions describes the swipe target widget or back-button element).
    The LLM is instructed to reason appropriately for all action types.
    """
    triplet_reasoning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant specialized in understanding and reasoning "
                    "about UI operation chains. You need to analyze the given "
                    "page-element-page triplet information and perform deep understanding "
                    "and reasoning. The action may be any UI interaction type: tap, text "
                    "input, long press, swipe, or back navigation. For swipe and back "
                    "actions the element describes the interaction target (e.g. the area "
                    "swiped over or the back button). You will receive textual descriptions "
                    "and screenshots of pages — please analyze both."
                ),
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the following UI operation triplet:\n\n"
                            "  Source Page : {source_page_desc}\n"
                            "  Element     : {element_desc}\n"
                            "  Action      : {action_name}\n"
                            "  Target Page : {target_page_desc}\n\n"
                            "Please reason about the following aspects:\n"
                            "1. What is the context and purpose of this operation?\n"
                            "2. What might be the user's intention when performing this operation?\n"
                            "3. How does this operation affect the page state change?\n"
                            "4. What is the relationship between this operation and the overall task flow?\n"
                            "5. Based on your understanding, generate richer and more accurate "
                            "descriptions for the source page, element, and target page.\n\n"
                            "Return your reasoning as a structured JSON object with the fields:\n"
                            "- context: Operation context description\n"
                            "- user_intent: User intention analysis\n"
                            "- state_change: State change description\n"
                            "- task_relation: Relationship with the task\n"
                            "- source_page_enhanced_desc: Enhanced description of source page\n"
                            "- element_enhanced_desc: Enhanced description of element\n"
                            "- target_page_enhanced_desc: Enhanced description of target page\n\n"
                            "{format_instructions}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{source_page_image}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{target_page_image}"},
                    },
                ],
            ),
        ]
    )

    parser = JsonOutputParser(pydantic_object=TripletReasoning)
    prompt = triplet_reasoning_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    return RunnablePassthrough() | prompt | model | parser


def create_merge_descriptions_chain():
    """Create LCEL chain for merging page descriptions."""
    merge_descriptions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant specialized in merging and optimizing page "
                    "descriptions. You need to analyze descriptions of shared pages between "
                    "two adjacent triplets and merge them into a more complete description."
                ),
            ),
            (
                "human",
                (
                    "Please analyze the following two descriptions that describe the same "
                    "page but from different contexts:\n\n"
                    "Current Task: {task_info}\n\n"
                    "Description 1 (as target page of previous triplet): {desc1}\n"
                    "Description 2 (as source page of next triplet): {desc2}\n\n"
                    "Please merge these two descriptions to generate a more complete and "
                    "coherent description. Requirements:\n"
                    "1. Consider the context and goals of the current task\n"
                    "2. Preserve all important information\n"
                    "3. Eliminate redundant content\n"
                    "4. Ensure description coherence\n"
                    "5. Highlight core functionality and features of the page\n"
                    "6. Emphasize relevance to the current task\n\n"
                    "Please return the merged description."
                ),
            ),
        ]
    )
    return RunnablePassthrough() | merge_descriptions_prompt | model | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
#  Core processing functions
# ─────────────────────────────────────────────────────────────────────────────

async def process_triplet(triplet: Dict[str, Any], reasoning_chain) -> Dict[str, Any]:
    """
    Process reasoning for a single triplet.

    Works uniformly for ALL action types (tap, text, long_press, swipe, back).
    graph_db guarantees that every hop now has:
      - triplet["element"]     : a dict (never None)
      - triplet["action"]["action_name"] : a non-empty string

    The _resolve_* helpers below add a final safety net for any legacy data
    that predates the graph_db update.

    Args:
        triplet: Hop dict from get_chain_from_start()
        reasoning_chain: LCEL triplet reasoning chain

    Returns:
        The same triplet dict enriched with a "reasoning" key, and with
        source_page, element, and target_page descriptions updated in-place.
    """
    # ── Resolve element and action_name safely ────────────────────────────────
    element = _resolve_element(triplet)
    action_name = _resolve_action_name(triplet)

    # ── Build text inputs for the LLM ─────────────────────────────────────────
    reasoning_input: Dict[str, Any] = {
        "source_page_desc": triplet["source_page"].get("description", ""),
        "element_desc":     element.get("description", ""),
        "target_page_desc": triplet["target_page"].get("description", ""),
        "action_name":      action_name,
    }

    # ── Load page screenshots ─────────────────────────────────────────────────
    try:
        source_page_image = "data:image/png;base64,"
        if triplet["source_page"].get("raw_page_url"):
            source_page_image = _load_image_to_base64(
                triplet["source_page"]["raw_page_url"]
            )

        target_page_image = "data:image/png;base64,"
        if triplet["target_page"].get("raw_page_url"):
            target_page_image = _load_image_to_base64(
                triplet["target_page"]["raw_page_url"]
            )

        reasoning_input["source_page_image"] = source_page_image
        reasoning_input["target_page_image"] = target_page_image
    except Exception as e:
        print(f"Error loading page images: {e}")
        reasoning_input["source_page_image"] = "data:image/png;base64,"
        reasoning_input["target_page_image"] = "data:image/png;base64,"

    # ── Run reasoning chain ───────────────────────────────────────────────────
    if _LLM_ACCESS_DENIED:
        triplet["reasoning_error"] = "LLM access denied (403); reasoning skipped"
        return triplet

    try:
        await wait_for_llm_slot()
        reasoning_result = await reasoning_chain.ainvoke(reasoning_input)

        triplet["reasoning"] = reasoning_result

        # Write enhanced descriptions back into the triplet in-place
        triplet["source_page"]["description"] = reasoning_result[
            "source_page_enhanced_desc"
        ]
        # Always write back via the resolved element dict — which IS the same
        # object as triplet["element"] when element is not None, or is a new
        # placeholder dict when it was None.  Ensure triplet["element"] points
        # to the resolved dict so the DB update loop below finds it.
        element["description"] = reasoning_result["element_enhanced_desc"]
        triplet["element"] = element
        triplet["target_page"]["description"] = reasoning_result[
            "target_page_enhanced_desc"
        ]

    except Exception as e:
        if _is_access_denied_error(e):
            _mark_llm_access_denied(e)
        print(
            f"Error during triplet reasoning "
            f"(src={triplet['source_page'].get('page_id', '?')} "
            f"action={action_name} "
            f"tgt={triplet['target_page'].get('page_id', '?')}): {e}"
        )
        _print_exception_details("triplet_reasoning", e)
        triplet["reasoning_error"] = str(e)

    return triplet


async def merge_node_descriptions(
    chain: List[Dict[str, Any]], merge_chain, task_info: str
) -> List[Dict[str, Any]]:
    """
    Merge descriptions of overlapping page nodes in adjacent triplets.

    When triplet[i].target_page and triplet[i+1].source_page share the same
    page_id, their descriptions are independently enriched by reasoning and
    may contain complementary information.  This step merges them into a
    single coherent description and writes it back to both slots.
    """
    for i in range(len(chain) - 1):
        if _LLM_ACCESS_DENIED:
            break

        current = chain[i]
        nxt = chain[i + 1]

        if (
            current["target_page"].get("page_id")
            == nxt["source_page"].get("page_id")
        ):
            merge_input = {
                "desc1":     current["target_page"].get("description", ""),
                "desc2":     nxt["source_page"].get("description", ""),
                "task_info": task_info,
            }
            try:
                await wait_for_llm_slot()
                merged_desc = await merge_chain.ainvoke(merge_input)
                current["target_page"]["description"] = merged_desc
                nxt["source_page"]["description"] = merged_desc
            except Exception as e:
                if _is_access_denied_error(e):
                    _mark_llm_access_denied(e)
                print(f"Error merging descriptions at step {i}: {e}")
                _print_exception_details(f"merge_step_{i}", e)

    return chain


def update_node_in_db(
    node_id: str,
    property_name: str,
    property_value: Any,
    node_type: Optional[str] = None,
) -> bool:
    """Update a single property on a node in Neo4j."""
    try:
        return db.update_node_property(
            node_id=node_id,
            property_name=property_name,
            property_value=property_value,
            node_type=node_type,
        )
    except Exception as e:
        print(f"Error updating node property: {e}")
        return False


async def process_single_chain(
    chain: List[Dict[str, Any]], reasoning_chain, merge_chain
) -> List[Dict[str, Any]]:
    """
    Process all triplets in a chain and persist results to Neo4j.

    Steps:
      1. Extract task description from the first hop's source page.
      2. Run element-based triplet reasoning on every hop (tap, text,
         long_press, swipe, back — all treated identically).
      3. Merge overlapping page descriptions between adjacent hops.
      4. Write enriched descriptions and reasoning back to Neo4j.

    Args:
        chain: List of hop dicts from get_chain_from_start()
        reasoning_chain: Triplet reasoning LCEL chain
        merge_chain: Description merging LCEL chain

    Returns:
        The fully processed chain with enriched descriptions in every hop.
    """
    if not chain:
        print("Warning: process_single_chain received an empty chain")
        return []

    # ── 1. Extract task info ──────────────────────────────────────────────────
    task_info = "Unknown Task"
    try:
        other_info = chain[0]["source_page"].get("other_info", {})
        if isinstance(other_info, str):
            other_info = json.loads(other_info)
        task_info = (
            other_info.get("task_info", {}).get("description", "Unknown Task")
        )
        print(f"Extracted task information: {task_info}")
    except Exception as e:
        print(f"Error extracting task information: {e}")

    # ── 2. Reason over every triplet ──────────────────────────────────────────
    print(f"Processing {len(chain)} triplet(s)...")
    processed_triplets = []
    for idx, triplet in enumerate(chain):
        hop_type = triplet.get("hop_type", "unknown")
        action_name = _resolve_action_name(triplet)
        print(
            f"  [{idx + 1}/{len(chain)}] hop_type={hop_type} action={action_name} "
            f"src={triplet['source_page'].get('page_id', '?')} "
            f"tgt={triplet['target_page'].get('page_id', '?')}"
        )
        processed_triplet = await process_triplet(triplet, reasoning_chain)
        processed_triplets.append(processed_triplet)

    # ── 3. Merge overlapping page descriptions ────────────────────────────────
    merged_chain = await merge_node_descriptions(
        processed_triplets, merge_chain, task_info
    )

    # ── 4. Persist to Neo4j ───────────────────────────────────────────────────
    for triplet in merged_chain:
        # Update source page description
        update_node_in_db(
            triplet["source_page"]["page_id"],
            "description",
            triplet["source_page"].get("description", ""),
            "Page",
        )

        # Update target page description
        update_node_in_db(
            triplet["target_page"]["page_id"],
            "description",
            triplet["target_page"].get("description", ""),
            "Page",
        )

        # Update element description
        # _resolve_element guarantees triplet["element"] is always a dict.
        # Only write to DB if we have a real element_id.
        element = triplet["element"]
        element_id = element.get("element_id", "")
        if element_id:
            update_node_in_db(
                element_id,
                "description",
                element.get("description", ""),
                "Element",
            )

            # Save complete reasoning results to the element node
            if "reasoning" in triplet:
                update_node_in_db(
                    element_id,
                    "reasoning",
                    json.dumps(triplet["reasoning"]),
                    "Element",
                )
        else:
            print(
                f"  Skipping DB element write for hop "
                f"src={triplet['source_page'].get('page_id', '?')} — "
                f"no element_id (legacy direct hop)"
            )

    return merged_chain


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def process_and_update_chain(start_page_id: str) -> List[Dict[str, Any]]:
    """
    Process a triplet chain starting from start_page_id and update Neo4j.

    Any page that has been stored via Tab ③ (Store to DB) and has a valid
    chain in Neo4j can be used here — regardless of whether the chain
    contains tap, text, long_press, swipe, or back actions.

    Args:
        start_page_id: The page_id of the first page in the recorded chain.
                       Use db.get_chain_start_nodes() to discover valid IDs.

    Returns:
        List of processed triplet dicts with enriched descriptions.
    """
    reasoning_chain = create_triplet_reasoning_chain()
    merge_chain = create_merge_descriptions_chain()

    triplets = db.get_chain_from_start(start_page_id)

    if not triplets:
        print(f"Warning: No triplets found for start_page_id={start_page_id!r}")
        return []

    print(f"Retrieved {len(triplets)} triplet(s) for start_page_id={start_page_id!r}")
    processed_chain = await process_single_chain(triplets, reasoning_chain, merge_chain)
    return processed_chain