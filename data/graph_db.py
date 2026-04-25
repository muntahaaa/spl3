import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class Neo4jDatabase:
    """Graph-storage adapter for human-exploration traces."""

    def __init__(self, uri: str, auth: tuple, database: str = "graphdb"):
        self.driver = None
        self.database = database
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            if uri.startswith("neo4j://") and "routing information" in str(exc).lower():
                fallback_uri = "bolt://" + uri[len("neo4j://"):]
                self.driver = GraphDatabase.driver(fallback_uri, auth=auth)
                self.driver.verify_connectivity()
            else:
                raise

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # ── node writes ───────────────────────────────────────────────────────────

    def create_page(self, properties: Dict[str, Any]) -> Optional[str]:
        """
        CAUSE-1 FIX: MERGE on page_id, then SET all other properties.
        Idempotent — re-running with the same page_id updates rather than
        duplicates.
        """
        required_fields = ["page_id"]
        if not all(f in properties for f in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties = dict(properties)   # don't mutate caller's dict
        properties.setdefault("timestamp", int(datetime.now().timestamp()))

        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        # CAUSE-1 FIX: MERGE on business key, SET the rest
        query = """
        MERGE (n:Page {page_id: $page_id})
        SET n += $props
        RETURN elementId(n) as node_id
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query,
                                 page_id=properties["page_id"],
                                 props=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    def create_element(self, properties: Dict[str, Any]) -> Optional[str]:
        """
        CAUSE-1 FIX: MERGE on element_id, then SET all other properties.
        """
        required_fields = ["element_id"]
        if not all(f in properties for f in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties = dict(properties)
        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])
        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        # CAUSE-1 FIX: MERGE on business key
        query = """
        MERGE (n:Element {element_id: $element_id})
        SET n += $props
        RETURN elementId(n) as node_id
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query,
                                 element_id=properties["element_id"],
                                 props=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    def create_action(self, properties: Dict[str, Any]) -> Optional[str]:
        """
        CAUSE-1 FIX: MERGE on action_id, then SET all other properties.
        """
        required_fields = ["action_id"]
        if not all(f in properties for f in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties = dict(properties)
        properties.setdefault("timestamp", int(datetime.now().timestamp()))
        if "action_result" in properties:
            if isinstance(properties["action_result"], dict):
                properties["action_result"] = json.dumps(properties["action_result"])

        # CAUSE-1 FIX: MERGE on business key
        query = """
        MERGE (n:Action {action_id: $action_id})
        SET n += $props
        RETURN elementId(n) as node_id
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query,
                                 action_id=properties["action_id"],
                                 props=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    def update_node_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
        node_type: Optional[str] = None,
    ) -> bool:
        if isinstance(property_value, (dict, list)):
            property_value = json.dumps(property_value)

        if node_type:
            id_field = node_type.lower() + "_id"
            query = f"""
            MATCH (n:{node_type})
            WHERE n.{id_field} = $node_id
            SET n[$property_name] = $property_value
            RETURN n
            """
        else:
            query = """
            MATCH (n)
            WHERE n.page_id = $node_id
               OR n.element_id = $node_id
               OR n.action_id = $node_id
            SET n[$property_name] = $property_value
            RETURN n
            """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    node_id=node_id,
                    property_value=property_value,
                    property_name=property_name,
                )
                record = result.single()
                if not record:
                    print(f"Warning: update_node_property: node {node_id} not found")
                return record is not None
        except Exception as exc:
            print(f"Error updating node property: {exc}")
            return False

    # ── relationships ─────────────────────────────────────────────────────────

    def add_element_to_page(self, page_id: str, element_id: str) -> bool:
        """HAS_ELEMENT: Page → Element. Already uses MERGE — no change needed."""
        query = """
        MATCH (p:Page    {page_id:    $page_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (p)-[r:HAS_ELEMENT]->(e)
        RETURN type(r) as rel_type
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, page_id=page_id, element_id=element_id)
                record = result.single()
                if not record:
                    print(f"Warning: HAS_ELEMENT failed: page={page_id} elem={element_id}")
                return record is not None
        except Exception as exc:
            print(f"Error creating HAS_ELEMENT: {exc}")
            return False

    def add_element_to_action(
        self,
        action_id: str,
        element_id: str,
        order: int,
        atomic_action: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        CAUSE-3 FIX for COMPOSED_OF:
        MERGE on the stable structural key (action_id, element_id, order).
        SET the mutable properties (atomic_action, action_params) separately
        so re-runs update rather than duplicate.
        """
        if action_params:
            action_params = json.dumps(action_params)

        # CAUSE-3 FIX: split MERGE key from SET properties
        query = """
        MATCH (a:Action  {action_id:  $action_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (a)-[r:COMPOSED_OF {order: $order}]->(e)
        SET r.atomic_action = $atomic_action,
            r.action_params  = $action_params
        RETURN type(r) as rel_type
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    action_id=action_id,
                    element_id=element_id,
                    order=order,
                    atomic_action=atomic_action,
                    action_params=action_params or "",
                )
                return result.single() is not None
        except Exception as exc:
            print(f"Error creating COMPOSED_OF: {exc}")
            return False

    def add_element_leads_to(
        self,
        element_id: str,
        target_id: str,
        action_name: str,
        action_params: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
    ) -> bool:
        """
        CAUSE-3 FIX for LEADS_TO:
        MERGE on the stable key (element_id → page_id + action_name).
        SET action_params and confidence_score separately.
        """
        if action_params:
            action_params = json.dumps(action_params)

        # CAUSE-3 FIX: only stable fields in MERGE key
        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (t:Page    {page_id:    $target_id})
        MERGE (e)-[r:LEADS_TO {action_name: $action_name}]->(t)
        SET r.action_params    = $action_params,
            r.confidence_score = $confidence_score
        RETURN type(r) as rel_type
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    element_id=element_id,
                    target_id=target_id,
                    action_name=action_name,
                    action_params=action_params or "",
                    confidence_score=confidence_score,
                )
                record = result.single()
                if not record:
                    print(f"Warning: LEADS_TO failed: elem={element_id} page={target_id}")
                return record is not None
        except Exception as exc:
            print(f"Error creating LEADS_TO: {exc}")
            return False

    # ── read helpers ──────────────────────────────────────────────────────────

    def get_page_elements(self, page_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
        RETURN e
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, page_id=page_id)
            elements = []
            for record in result:
                element = dict(record["e"])
                if "possible_actions" in element:
                    try:
                        element["possible_actions"] = json.loads(element["possible_actions"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                elements.append(element)
            return elements

    def get_action_sequence(self, action_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a:Action {action_id: $action_id})-[r:COMPOSED_OF]->(e:Element)
        RETURN e.element_id  as element_id,
               e.element_type as element_type,
               r.order         as order,
               r.atomic_action as atomic_action,
               r.action_params as action_params
        ORDER BY r.order
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, action_id=action_id)
            sequences = []
            for record in result:
                record_dict = {
                    "element_id":    record["element_id"],
                    "element_type":  record["element_type"],
                    "order":         record["order"],
                    "atomic_action": record["atomic_action"],
                    "action_params": record["action_params"],
                }
                if record_dict.get("action_params"):
                    try:
                        record_dict["action_params"] = json.loads(record_dict["action_params"])
                    except json.JSONDecodeError:
                        pass
                sequences.append(record_dict)
            return sequences



    # ─────────────────────────────────────────────────────────────────────────
    #  get_chain_start_nodes
    # ─────────────────────────────────────────────────────────────────────────
    def get_chain_start_nodes(self) -> List[Dict[str, Any]]:
        """
        Returns genuine start pages only.

                A start page satisfies ALL of:
                    1. No incoming LEADS_TO edge     (no element navigated TO it)
                    2. Has at least one HAS_ELEMENT  (it is a real parsed page)
        """
        query = """
        MATCH (n:Page)
                WHERE NOT EXISTS { ()-[:LEADS_TO]->(n) }
                    AND EXISTS     { (n)-[:HAS_ELEMENT]->() }
        RETURN n
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                return [dict(record["n"]) for record in result]
        except Exception as exc:
            print(f"Error getting chain start nodes: {exc}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  get_chain_from_start  — element-based triplets only
    # ─────────────────────────────────────────────────────────────────────────
    def get_chain_from_start(self, start_page_id: str) -> List[Dict[str, Any]]:
        """
        Walk the full exploration chain from start_page_id and return a flat
        list of element-based hop dicts.

            {
              "source_page" : {page properties},
              "target_page" : {page properties},
              "element"     : {element properties},   # always a dict, never None
              "action"      : {                        # normalised action dict
                  "action_name"  : str,               # unified key for all types
                  "action_type"  : str,               # original type label
                  "action_params": str,
                  ...other relationship properties
              },
              "hop_type"    : "element_hop" | "direct_hop",
            }

        HOW ELEMENT HOPS WORK (tap / text / long_press)
        ─────────────────────────────────────────────────
        Path: (Page)-[:HAS_ELEMENT]->(Element)-[:LEADS_TO]->(Page)
        The element is the node that was interacted with.
        action_name comes from the LEADS_TO.action_name property.

        """
        # ── Element-mediated hops (tap / text / long_press / swipe / back) ───
        elem_query = """
        MATCH (start:Page {page_id: $start_page_id})
        MATCH (src:Page)-[:HAS_ELEMENT]->(e:Element)-[lt:LEADS_TO]->(tgt:Page)
        WHERE src.page_id = start.page_id
           OR EXISTS {
               MATCH (start)-[:HAS_ELEMENT|LEADS_TO*]->(src)
           }
        RETURN src, e, lt, tgt
        """

        def _to_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        try:
            with self.driver.session(database=self.database) as session:

                elem_hops = []
                for record in session.run(elem_query, start_page_id=start_page_id):
                    lt = dict(record["lt"])
                    # Normalise: LEADS_TO already has action_name; add action_type alias
                    lt.setdefault("action_type", lt.get("action_name", ""))

                    elem_hops.append({
                        "source_page": dict(record["src"]),
                        "target_page": dict(record["tgt"]),
                        "element":     dict(record["e"]),
                        "action":      lt,
                        "hop_type":    "element_hop",
                        "_sort_key":   _to_int(record["tgt"].get("timestamp", 0)),
                    })

            # ── Sort, deduplicate ────────────────────────────────────────────
            elem_hops.sort(key=lambda h: h["_sort_key"])

            seen: set = set()
            chain: List[Dict[str, Any]] = []
            for hop in elem_hops:
                key = (
                    hop["source_page"].get("page_id"),
                    hop["target_page"].get("page_id"),
                )
                if key not in seen:
                    seen.add(key)
                    hop.pop("_sort_key")
                    chain.append(hop)

            return chain

        except Exception as exc:
            print(f"Error getting chain from start: {exc}")
            return []

    # ── read helpers (ported from graph_db2) ─────────────────────────────────

    def get_all_actions(self) -> List[Dict[str, Any]]:
        """Get all Action nodes from the database."""
        query = """
        MATCH (a:Action)
        RETURN a
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    if "element_sequence" in action and isinstance(action["element_sequence"], str):
                        try:
                            action["element_sequence"] = json.loads(action["element_sequence"])
                        except json.JSONDecodeError:
                            pass
                    actions.append(action)
                return actions
        except Exception as exc:
            print(f"Error getting all actions: {exc}")
            return []

    def get_action_by_id(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get Action node by ID, or None if not found."""
        query = """
        MATCH (a:Action)
        WHERE a.action_id = $action_id
        RETURN a
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, action_id=action_id)
                record = result.single()
                if record:
                    action = dict(record["a"])
                    if "element_sequence" in action and isinstance(action["element_sequence"], str):
                        try:
                            action["element_sequence"] = json.loads(action["element_sequence"])
                        except json.JSONDecodeError:
                            pass
                    return action
                return None
        except Exception as exc:
            print(f"Error getting action by ID {action_id}: {exc}")
            return None

    def get_element_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get Element node by ID, or None if not found."""
        query = """
        MATCH (e:Element)
        WHERE e.element_id = $element_id
        RETURN e
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, element_id=element_id)
                record = result.single()
                if record:
                    element = dict(record["e"])
                    for field in ("possible_actions", "other_info"):
                        if field in element and isinstance(element[field], str):
                            try:
                                element[field] = json.loads(element[field])
                            except json.JSONDecodeError:
                                pass
                    return element
                return None
        except Exception as exc:
            print(f"Error getting element by ID {element_id}: {exc}")
            return None

    def get_all_high_level_actions(self) -> List[Dict[str, Any]]:
        """Get all high-level Action nodes (is_high_level, high_level, or type='high_level')."""
        query = """
        MATCH (a:Action)
        WHERE a.is_high_level = true OR a.high_level = true OR a.type = 'high_level'
        RETURN a
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    if "element_sequence" in action and isinstance(action["element_sequence"], str):
                        try:
                            action["element_sequence"] = json.loads(action["element_sequence"])
                        except json.JSONDecodeError:
                            pass
                    actions.append(action)
                return actions
        except Exception as exc:
            print(f"Error getting all high level actions: {exc}")
            return []

    def get_high_level_actions_for_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Get high-level Action nodes whose name or description contains the task string.
        Falls back to get_all_high_level_actions() when no match is found.
        """
        query = """
        MATCH (a:Action)
        WHERE (a.is_high_level = true OR a.high_level = true OR a.type = 'high_level')
          AND (a.name CONTAINS $task OR a.description CONTAINS $task)
        RETURN a
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, task=task)
                actions = []
                for record in result:
                    action = dict(record["a"])
                    if "element_sequence" in action and isinstance(action["element_sequence"], str):
                        try:
                            action["element_sequence"] = json.loads(action["element_sequence"])
                        except json.JSONDecodeError:
                            pass
                    actions.append(action)
            if not actions:
                return self.get_all_high_level_actions()
            return actions
        except Exception as exc:
            print(f"Error getting high level actions for task '{task}': {exc}")
            return []

    def get_shortcuts_for_action(self, action_id: str) -> List[Dict[str, Any]]:
        """Get Shortcut nodes connected via REFERS_TO to the given Action."""
        query = """
        MATCH (s:Shortcut)-[:REFERS_TO]->(a:Action {action_id: $action_id})
        RETURN s
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, action_id=action_id)
                shortcuts = []
                for record in result:
                    shortcut = dict(record["s"])
                    for field in ("conditions", "page_flow"):
                        if field in shortcut and isinstance(shortcut[field], str):
                            try:
                                shortcut[field] = json.loads(shortcut[field])
                            except json.JSONDecodeError:
                                pass
                    shortcuts.append(shortcut)
                return shortcuts
        except Exception as exc:
            print(f"Error getting shortcuts for action '{action_id}': {exc}")
            return []

    def get_page_by_visual_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Get Page node by visual_embedding_id, or None if not found."""
        query = """
        MATCH (p:Page)
        WHERE p.visual_embedding_id = $embedding_id
        RETURN p
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, embedding_id=embedding_id)
                record = result.single()
                if record:
                    page = dict(record["p"])
                    for field in ("elements_data", "metadata"):
                        if field in page and isinstance(page[field], str):
                            try:
                                page[field] = json.loads(page[field])
                            except json.JSONDecodeError:
                                pass
                    return page
                return None
        except Exception as exc:
            print(f"Error getting page by visual embedding ID {embedding_id}: {exc}")
            return None