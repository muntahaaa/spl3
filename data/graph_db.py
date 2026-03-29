import json
from datetime import datetime
from typing import Any, Dict

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
        self.driver.close()

    # ── generic node creation ─────────────────

    def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        query = f"CREATE (n:{label} $properties) RETURN elementId(n) AS node_id"
        with self.driver.session(database=self.database) as session:
            record = session.run(query, properties=properties).single()
            return str(record["node_id"]) if record else ""

    # ── typed helpers ─────────────────────────

    def create_page(self, properties: Dict[str, Any]) -> str:
        props = dict(properties)
        props.setdefault("timestamp", int(datetime.now().timestamp()))
        return self.create_node("Page", props)

    def create_element(self, properties: Dict[str, Any]) -> str:
        return self.create_node("Element", dict(properties))

    # ── relationships ─────────────────────────

    def add_element_to_page(self, page_id: str, element_id: str) -> bool:
        query = """
        MATCH (p:Page    {page_id:    $page_id})
        MATCH (e:Element {element_id: $element_id})
        MERGE (p)-[:HAS_ELEMENT]->(e)
        RETURN 1 AS ok
        """
        with self.driver.session(database=self.database) as session:
            return session.run(query, page_id=page_id, element_id=element_id).single() is not None

    def add_element_leads_to(
        self,
        element_id: str,
        target_page_id: str,
        action_name: str,
        action_params: Dict[str, Any],
    ) -> bool:
        # Neo4j properties cannot contain nested maps, so keep primitives
        # directly and store the full params payload as JSON text.
        execution_result = action_params.get("execution_result", "")
        action_timestamp = action_params.get("timestamp", "")
        action_params_json = json.dumps(action_params, ensure_ascii=False)

        query = """
        MATCH (e:Element {element_id: $element_id})
        MATCH (p:Page    {page_id:    $target_page_id})
        MERGE (e)-[:LEADS_TO {
            action_name:       $action_name,
            execution_result:  $execution_result,
            timestamp:         $action_timestamp,
            action_params_json:$action_params_json
        }]->(p)
        RETURN 1 AS ok
        """
        with self.driver.session(database=self.database) as session:
            return (
                session.run(
                    query,
                    element_id=element_id,
                    target_page_id=target_page_id,
                    action_name=action_name,
                    execution_result=execution_result,
                    action_timestamp=action_timestamp,
                    action_params_json=action_params_json,
                ).single()
                is not None
            )
