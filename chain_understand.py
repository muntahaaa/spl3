"""
chain_understand.py  (NVIDIA NIM edition)
-----------------------------------------
Replaces Firebase round-trips with direct NVIDIA NIM API calls
(nvidia/llama-3.1-nemotron-nano-vl-8b-v1) via the OpenAI-compatible client.

Public API is unchanged:
    await process_and_update_chain(start_page_id)  →  List[Dict]
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import io
import time
import traceback
from typing import Any, Dict, List, Optional
from PIL import Image 

import config
from data.graph_db import Neo4jDatabase
from llm_rate_limit import wait_for_llm_slot
from nvidia_llm_bridge import NvidiaBridge

# ── LangSmith tracing (kept for observability) ───────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.LANGCHAIN_TRACING_V2 else "false"
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "ChainEvolve"

# ── Database ─────────────────────────────────────────────────────────────────
db = Neo4jDatabase(config.Neo4j_URI, config.Neo4j_AUTH, database=config.Neo4j_DB)

# ── NVIDIA NIM bridge (direct — no Firebase worker needed) ────────────────────
bridge = NvidiaBridge()

_LLM_ACCESS_DENIED = False


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_action_name(triplet: Dict[str, Any]) -> str:
    action = triplet.get("action") or {}
    return action.get("action_name") or action.get("action_type") or ""


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


def _load_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """Return raw base64 string (no data-URL prefix) for an image file."""
    if not image_path or not os.path.exists(image_path):
        return ""
    with Image.open(image_path) as img:
        # Resize so longest edge ≤ max_size
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_access_denied_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "permissiondenied" in msg or (
        "403" in msg and ("denied access" in msg or "permission denied" in msg)
    )


def _mark_llm_access_denied(exc: Exception) -> None:
    global _LLM_ACCESS_DENIED
    if not _LLM_ACCESS_DENIED:
        _LLM_ACCESS_DENIED = True
        print(f"LLM access denied. Skipping further LLM calls for this run.\n{exc}")


def _print_exception_details(prefix: str, exc: Exception) -> None:
    print(f"{prefix} | {type(exc).__name__}: {repr(exc)}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builders (plain strings replacing LangChain prompt templates)
# ─────────────────────────────────────────────────────────────────────────────

_TRIPLET_SYSTEM = (
    "You are an AI assistant specialized in understanding and reasoning about UI "
    "operation chains. Analyse the given page-element-page triplet and perform deep "
    "reasoning. The action may be tap, text input, long press, swipe, or back. "
    "For swipe/back, the element describes the interaction target. "
    "You receive textual descriptions and optionally screenshots — analyse both.\n\n"
    "Return ONLY a JSON object with these exact keys:\n"
    "  context, user_intent, state_change, task_relation,\n"
    "  source_page_enhanced_desc, element_enhanced_desc, target_page_enhanced_desc\n"
    "No preamble, no markdown fences."
)


def _build_triplet_user_prompt(
    source_page_desc: str,
    element_desc: str,
    action_name: str,
    target_page_desc: str,
) -> str:
    return (
        f"Source Page : {source_page_desc}\n"
        f"Element     : {element_desc}\n"
        f"Action      : {action_name}\n"
        f"Target Page : {target_page_desc}\n\n"
        "Reason about:\n"
        "1. Context and purpose of this operation\n"
        "2. User intention\n"
        "3. Page state change\n"
        "4. Relationship with the overall task flow\n"
        "5. Richer descriptions for source page, element, target page\n\n"
        "Return JSON only."
    )


_MERGE_SYSTEM = (
    "You are an AI assistant that merges two descriptions of the same page "
    "(seen from different adjacent triplets) into one coherent description. "
    "Preserve all important information, eliminate redundancy, and highlight "
    "the page's core functionality. Return only the merged description text, "
    "no JSON, no markdown."
)


def _build_merge_user_prompt(desc1: str, desc2: str, task_info: str) -> str:
    return (
        f"Current Task: {task_info}\n\n"
        f"Description 1 (as target page of previous triplet): {desc1}\n"
        f"Description 2 (as source page of next triplet): {desc2}\n\n"
        "Merge these into a single coherent description."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Core async processing
# ─────────────────────────────────────────────────────────────────────────────

async def process_triplet(triplet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run LLM reasoning over a single (source_page, element, target_page) hop.
    All action types are handled uniformly.
    """
    global _LLM_ACCESS_DENIED

    element     = _resolve_element(triplet)
    action_name = _resolve_action_name(triplet)
    src_id      = triplet['source_page'].get('page_id', '?')
    tgt_id      = triplet['target_page'].get('page_id', '?')

    print(f"    [triplet] Building prompt for src={src_id[:8]} action={action_name} tgt={tgt_id[:8]}")


    user_prompt = _build_triplet_user_prompt(
        source_page_desc=triplet["source_page"].get("description", ""),
        element_desc=element.get("description", ""),
        action_name=action_name,
        target_page_desc=triplet["target_page"].get("description", ""),
    )

    # Collect page screenshots as base64 images (load both concurrently)
    async def _load_page_image(page_key: str) -> str:
        img_path = triplet[page_key].get("raw_page_url", "")
        print(f"    [triplet] Loading image for {page_key}: {img_path or '(none)'}")
        b64 = await asyncio.to_thread(_load_image_to_base64, img_path)
        if b64:
            print(f"    [triplet] Image loaded OK ({len(b64)//1024} KB base64)")
        else:
            print(f"    [triplet] No image found for {page_key}")
        return b64

    raw_images = await asyncio.gather(
        _load_page_image("source_page"),
        _load_page_image("target_page"),
    )
    images_b64: List[str] = [b for b in raw_images if b]

    if _LLM_ACCESS_DENIED:
        triplet["reasoning_error"] = "LLM access denied; reasoning skipped"
        return triplet

    print(f"    [triplet] Waiting for LLM slot...")
    try:
        await wait_for_llm_slot()
        print(f"    [triplet] Calling NVIDIA NIM (images={len(images_b64)})...")
        reasoning_result = await bridge.call_json(
            system_prompt=_TRIPLET_SYSTEM,
            user_prompt=user_prompt,
            images_b64=images_b64 if images_b64 else None,
        )
        print(f"    [triplet] Got result from NVIDIA NIM OK")

        triplet["reasoning"] = reasoning_result

        triplet["source_page"]["description"] = reasoning_result.get(
            "source_page_enhanced_desc", triplet["source_page"].get("description", "")
        )
        element["description"] = reasoning_result.get(
            "element_enhanced_desc", element.get("description", "")
        )
        triplet["element"] = element
        triplet["target_page"]["description"] = reasoning_result.get(
            "target_page_enhanced_desc", triplet["target_page"].get("description", "")
        )

    except Exception as e:
        if _is_access_denied_error(e):
            _mark_llm_access_denied(e)
        print(
            f"Error during triplet reasoning "
            f"(src={triplet['source_page'].get('page_id','?')} "
            f"action={action_name} "
            f"tgt={triplet['target_page'].get('page_id','?')}): {e}"
        )
        _print_exception_details("triplet_reasoning", e)
        triplet["reasoning_error"] = str(e)

    return triplet


async def merge_node_descriptions(
    chain: List[Dict[str, Any]], task_info: str
) -> List[Dict[str, Any]]:
    """
    Merge overlapping page descriptions between adjacent triplets.
    triplet[i].target_page == triplet[i+1].source_page  →  merge.
    """
    global _LLM_ACCESS_DENIED

    for i in range(len(chain) - 1):
        if _LLM_ACCESS_DENIED:
            break

        current = chain[i]
        nxt     = chain[i + 1]

        if (
            current["target_page"].get("page_id")
            != nxt["source_page"].get("page_id")
        ):
            continue

        user_prompt = _build_merge_user_prompt(
            desc1=current["target_page"].get("description", ""),
            desc2=nxt["source_page"].get("description", ""),
            task_info=task_info,
        )

        try:
            await wait_for_llm_slot()
            merged_desc = await bridge.call_text(
                system_prompt=_MERGE_SYSTEM,
                user_prompt=user_prompt,
            )
            current["target_page"]["description"] = merged_desc
            nxt["source_page"]["description"]     = merged_desc
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


async def process_single_chain(chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all triplets: reason → merge → persist.
    """
    if not chain:
        print("Warning: process_single_chain received an empty chain")
        return []

    # 1. Extract task info
    task_info = "Unknown Task"
    try:
        other_info = chain[0]["source_page"].get("other_info", {})
        if isinstance(other_info, str):
            other_info = json.loads(other_info)
        task_info = other_info.get("task_info", {}).get("description", "Unknown Task")
        print(f"Extracted task information: {task_info}")
    except Exception as e:
        print(f"Error extracting task information: {e}")

    # 2. Reason over every triplet (all concurrently — each waits for its own LLM slot)
    print(f"Processing {len(chain)} triplet(s) concurrently...")

    async def _process_with_log(idx: int, triplet: Dict[str, Any]) -> Dict[str, Any]:
        hop_type    = triplet.get("hop_type", "unknown")
        action_name = _resolve_action_name(triplet)
        print(
            f"  [{idx + 1}/{len(chain)}] hop_type={hop_type} action={action_name} "
            f"src={triplet['source_page'].get('page_id','?')} "
            f"tgt={triplet['target_page'].get('page_id','?')}"
        )
        return await process_triplet(triplet)

    processed_triplets: List[Dict[str, Any]] = list(
        await asyncio.gather(*[_process_with_log(i, t) for i, t in enumerate(chain)])
    )

    # 3. Merge overlapping page descriptions
    merged_chain = await merge_node_descriptions(processed_triplets, task_info)

    # 4. Persist to Neo4j (all writes batched concurrently)
    print(f"  [db] Persisting {len(merged_chain)} triplet(s) to Neo4j...")
    t0 = time.monotonic()

    async def _persist_triplet(triplet: Dict[str, Any]) -> None:
        element    = triplet["element"]
        element_id = element.get("element_id", "")
        writes = [
            ("source_page", triplet["source_page"]["page_id"], "description",
             triplet["source_page"].get("description", ""), "Page"),
            ("target_page", triplet["target_page"]["page_id"], "description",
             triplet["target_page"].get("description", ""), "Page"),
        ]
        if element_id:
            writes.append(("element", element_id, "description",
                           element.get("description", ""), "Element"))
            if "reasoning" in triplet:
                writes.append(("element_reasoning", element_id, "reasoning",
                               json.dumps(triplet["reasoning"]), "Element"))
        else:
            print(
                f"  Skipping DB element write for hop "
                f"src={triplet['source_page'].get('page_id','?')} — "
                f"no element_id (legacy direct hop)"
            )

        await asyncio.gather(*[
            asyncio.to_thread(update_node_in_db, node_id, prop, val, node_type)
            for _, node_id, prop, val, node_type in writes
        ])

    await asyncio.gather(*[_persist_triplet(t) for t in merged_chain])

    elapsed = time.monotonic() - t0
    print(f"  [db] Neo4j persist complete ({elapsed:.1f}s)")

    return merged_chain


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def process_and_update_chain(start_page_id: str) -> List[Dict[str, Any]]:
    """
    Process a triplet chain starting from start_page_id and update Neo4j.

    Args:
        start_page_id: The page_id of the first page in the recorded chain.

    Returns:
        List of processed triplet dicts with enriched descriptions.
    """
    triplets = db.get_chain_from_start(start_page_id)

    if not triplets:
        print(f"Warning: No triplets found for start_page_id={start_page_id!r}")
        return []

    print(f"Retrieved {len(triplets)} triplet(s) for start_page_id={start_page_id!r}")
    return await process_single_chain(triplets)