import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class Neo4jDatabase:
    """Minimal graph-storage adapter for human-exploration traces."""

    def __init__(self, uri: str, auth: tuple, database: str = "neo4j"):
        self.driver = None
        self.database = database
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            # Local single-node Neo4j often exposes bolt but not routing metadata.
            if uri.startswith("neo4j://") and "routing information" in str(exc).lower():
                fallback_uri = "bolt://" + uri[len("neo4j://"):]
                self.driver = GraphDatabase.driver(fallback_uri, auth=auth)
                self.driver.verify_connectivity()
            else:
                raise

    def close(self):
        if self.driver is not None:
            self.driver.close()

    # -- generic node creation -------------------------------------------------
    def create_action(self, properties: Dict[str, Any]) -> Optional[str]:
        """Create an Action node for high-level task operations."""
        required_fields = ["action_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")
    
        properties["timestamp"] = properties.get(
            "timestamp", int(datetime.now().timestamp())
        )
    
        if "action_result" in properties:
            if isinstance(properties["action_result"], dict):
                properties["action_result"] = json.dumps(properties["action_result"])
    
        return self.create_node("Action", properties)
    
    def update_node_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
        node_type: Optional[str] = None,  # "Page", "Element", "Action" or None
    ) -> bool:
        """Update a node property by node ID."""
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
            WHERE n.page_id = $node_id OR n.element_id = $node_id OR n.action_id = $node_id
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
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to update property {property_name} for node with ID {node_id}"
                    )
                return success
        except Exception as exc:
            print(f"Error updating node property: {exc}")
            return False

    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[str]:
        """Generic node creation function."""
        query = f"CREATE (n:{label} $properties) RETURN elementId(n) as node_id"

        with self.driver.session(database=self.database) as session:
            result = session.run(query, properties=properties)
            record = result.single()
            return str(record["node_id"]) if record else None

    # -- typed helpers ---------------------------------------------------------
    def create_page(self, properties: Dict[str, Any]) -> Optional[str]:
        required_fields = ["page_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        properties["timestamp"] = properties.get(
            "timestamp", int(datetime.now().timestamp())
        )

        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Page", properties)

    def create_element(self, properties: Dict[str, Any]) -> Optional[str]:
        required_fields = ["element_id"]
        if not all(field in properties for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        if "visual_embedding_id" in properties:
            properties["visual_embedding_id"] = str(properties["visual_embedding_id"])

        if "other_info" in properties:
            if isinstance(properties["other_info"], dict):
                properties["other_info"] = json.dumps(properties["other_info"])
            elif not isinstance(properties["other_info"], str):
                raise ValueError("other_info must be a dict or JSON string")

        return self.create_node("Element", properties)

    # -- relationships ---------------------------------------------------------
    def add_element_to_page(self, page_id: str, element_id: str) -> bool:
        query = """
        MATCH (p:Page {page_id: $page_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (p)-[r:HAS_ELEMENT]->(e)
        RETURN type(r) as rel_type
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, page_id=page_id, element_id=element_id)
                record = result.single()
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create HAS_ELEMENT relationship between page {page_id} and element {element_id}"
                    )
                return success
        except Exception as exc:
            print(f"Error creating HAS_ELEMENT relationship: {exc}")
            return False

    def add_element_to_action(
        self,
        action_id: str,
        element_id: str,
        order: int,
        atomic_action: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        query = """
        MATCH (a:Action {action_id: $action_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (a)-[r:COMPOSED_OF {
            order: $order,
            atomic_action: $atomic_action,
            action_params: $action_params
        }]->(e)
        RETURN type(r) as rel_type
        """

        if action_params:
            action_params = json.dumps(action_params)

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
                    element["possible_actions"] = json.loads(element["possible_actions"])
                elements.append(element)
            return elements

    def get_action_sequence(self, action_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (a:Action {action_id: $action_id})-[r:COMPOSED_OF]->(e:Element)
        RETURN e.element_id as element_id,
               e.element_type as element_type,
               r.order as order,
               r.atomic_action as atomic_action,
               r.action_params as action_params
        ORDER BY r.order
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, action_id=action_id)
            sequences = []
            for record in result:
                record_dict = {
                    "element_id": record["element_id"],
                    "element_type": record["element_type"],
                    "order": record["order"],
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

    def add_element_leads_to(
        self,
        element_id: str,
        target_id: str,
        action_name: str,
        action_params: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.0,
    ) -> bool:
        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (t:Page {page_id: $target_id})
        MERGE (e)-[r:LEADS_TO {
            action_name: $action_name,
            action_params: $action_params,
            confidence_score: $confidence_score
        }]->(t)
        RETURN type(r) as rel_type
        """

        try:
            if action_params:
                action_params = json.dumps(action_params)

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
                success = record is not None
                if not success:
                    print(
                        f"Warning: Failed to create LEADS_TO relationship from element {element_id} to page {target_id}"
                    )
                return success
        except Exception as exc:
            print(f"Error creating LEADS_TO relationship: {exc}")
            return False

    def get_chain_start_nodes(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (n:Page)
        WHERE NOT EXISTS { MATCH ()-[:LEADS_TO]->(n) }
        RETURN n
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                start_nodes = []
                for record in result:
                    start_nodes.append(dict(record["n"]))
                return start_nodes
        except Exception as exc:
            print(f"Error getting chain start nodes: {exc}")
            return []

    def get_chain_from_start(self, start_page_id: str) -> List[List[Dict[str, Any]]]:
        query = """
        MATCH path = (start:Page {page_id: $start_page_id})-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
        WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }
        WITH path, relationships(path) as rels, nodes(path) as nodes
        WITH DISTINCT [n in nodes | n{.*}] as node_props,
             [r in rels | r{.*}] as rel_props
        RETURN node_props, rel_props
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, start_page_id=start_page_id)
                chains = []
                seen_chains = set()

                for record in result:
                    nodes = record["node_props"]
                    rels = record["rel_props"]

                    chain = []
                    current_page = nodes[0]
                    i = 0

                    while i < len(rels):
                        if "element_id" in nodes[i + 1]:
                            element = nodes[i + 1]
                            if i + 1 < len(rels) and "action_name" in rels[i + 1]:
                                target_page = nodes[i + 2]
                                triplet = {
                                    "source_page": current_page,
                                    "element": element,
                                    "target_page": target_page,
                                    "action": rels[i + 1],
                                }
                                chain.append(triplet)
                                current_page = target_page
                                i += 2
                            else:
                                i += 1
                        else:
                            i += 1

                    if chain:
                        chain_key = tuple(
                            (
                                t["source_page"]["page_id"],
                                t["element"]["element_id"],
                                t["target_page"]["page_id"],
                            )
                            for t in chain
                        )
                        if chain_key not in seen_chains:
                            seen_chains.add(chain_key)
                            chains.append(chain)

                return chains[0] if chains else []
        except Exception as exc:
            print(f"Error getting chain from start node: {exc}")
            return []
