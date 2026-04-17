#!/usr/bin/env python3
"""
Neo4j Database Inspector - Execute Cypher queries and generate diagnostic report
Usage: python neo4j_inspector.py --uri bolt://localhost:7687 --user neo4j --password password
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


@dataclass
class QueryResult:
    """Store result of a Cypher query execution."""
    name: str
    query: str
    status: str  # "OK", "ERROR", "EMPTY"
    result: Any
    row_count: int
    error_message: Optional[str] = None
    execution_time: float = 0.0


class Neo4jInspector:
    """Execute diagnostic queries on Neo4j database."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = database
            self.results: List[QueryResult] = []
        except ServiceUnavailable as e:
            print(f"❌ Failed to connect to Neo4j at {uri}: {e}")
            sys.exit(1)

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def execute_query(self, name: str, query: str) -> QueryResult:
        """Execute a single Cypher query and return results."""
        start_time = datetime.now()
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                records = list(result)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if len(records) == 0:
                    status = "EMPTY"
                    result_data = []
                else:
                    status = "OK"
                    result_data = [dict(record) for record in records]
                
                return QueryResult(
                    name=name,
                    query=query,
                    status=status,
                    result=result_data,
                    row_count=len(records),
                    execution_time=execution_time
                )
        except Exception as e:
            return QueryResult(
                name=name,
                query=query,
                status="ERROR",
                result=None,
                row_count=0,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def run_health_check(self) -> Dict[str, Any]:
        """Run basic health check queries."""
        print("\n" + "="*80)
        print("NEO4J DATABASE HEALTH CHECK")
        print("="*80)

        # Query 1: Count nodes by type
        result1 = self.execute_query(
            "Node Count by Type",
            """
            MATCH (n)
            RETURN 
              labels(n)[0] as NodeType,
              count(n) as Count
            ORDER BY Count DESC
            """
        )
        self.results.append(result1)

        # Query 2: Relationship counts
        result2 = self.execute_query(
            "Relationship Count by Type",
            """
            MATCH ()-[r]->()
            RETURN 
              type(r) as RelationType,
              COUNT(r) as Count
            ORDER BY Count DESC
            """
        )
        self.results.append(result2)

        # Query 3: Database statistics
        result3 = self.execute_query(
            "Database Statistics",
            """
            CALL {
              MATCH (p:Page) RETURN COUNT(p) as PageCount
            }
            CALL {
              MATCH (e:Element) RETURN COUNT(e) as ElementCount
            }
            CALL {
              MATCH (a:Action) RETURN COUNT(a) as ActionCount
            }
            CALL {
              MATCH ()-[r:HAS_ELEMENT]->() RETURN COUNT(r) as HasElementCount
            }
            CALL {
              MATCH ()-[r:LEADS_TO]->() RETURN COUNT(r) as LeadsToCount
            }
            CALL {
              MATCH ()-[r:COMPOSED_OF]->() RETURN COUNT(r) as ComposedOfCount
            }
            RETURN 
              PageCount,
              ElementCount,
              ActionCount,
              HasElementCount,
              LeadsToCount,
              ComposedOfCount
            """
        )
        self.results.append(result3)

        # Query 4: Health status
        result4 = self.execute_query(
            "Database Health Status",
            """
            MATCH (n)
            WITH COUNT(n) as NodeCount
            MATCH ()-[r]->()
            RETURN 
              NodeCount as Nodes,
              COUNT(r) as Relationships,
              CASE 
                WHEN NodeCount = 0 AND COUNT(r) = 0 THEN 'EMPTY'
                WHEN NodeCount > 0 AND COUNT(r) = 0 THEN 'ORPHANED_NODES'
                WHEN NodeCount > 0 AND COUNT(r) > 0 THEN 'HEALTHY'
                ELSE 'UNKNOWN'
              END as DatabaseStatus
            """
        )
        self.results.append(result4)

        return self._format_results()

    def run_node_inspection(self) -> Dict[str, Any]:
        """Inspect all nodes in detail."""
        print("\n" + "="*80)
        print("NEO4J NODE INSPECTION")
        print("="*80)

        # Pages
        result1 = self.execute_query(
            "All Pages",
            """
            MATCH (p:Page)
            RETURN 
              p.page_id as PageId,
              p.description as Description,
              p.raw_page_url as Screenshot,
              p.timestamp as Timestamp
            ORDER BY p.timestamp ASC
            """
        )
        self.results.append(result1)

        # Pages with element counts
        result2 = self.execute_query(
            "Pages with Element Counts",
            """
            MATCH (p:Page)
            OPTIONAL MATCH (p)-[:HAS_ELEMENT]->(e:Element)
            RETURN 
              p.page_id as PageId,
              COUNT(e) as ElementCount
            ORDER BY p.timestamp ASC
            """
        )
        self.results.append(result2)

        # Elements (sample)
        result3 = self.execute_query(
            "Elements (first 20)",
            """
            MATCH (e:Element)
            RETURN 
              e.element_id as ElementId,
              e.element_original_id as OriginalId,
              e.action_type as ActionType,
              e.bounding_box as BBox,
              e.visual_embedding_id as EmbeddingId
            LIMIT 20
            """
        )
        self.results.append(result3)

        # Element statistics
        result4 = self.execute_query(
            "Element Count by Action Type",
            """
            MATCH (e:Element)
            RETURN 
              e.action_type as ActionType,
              COUNT(e) as Count
            ORDER BY Count DESC
            """
        )
        self.results.append(result4)

        # Actions
        result5 = self.execute_query(
            "All Action Nodes",
            """
            MATCH (a:Action)
            RETURN 
              a.action_id as ActionId,
              a.action_name as ActionName,
              a.step as Step,
              a.timestamp as Timestamp
            ORDER BY a.step ASC
            """
        )
        self.results.append(result5)

        return self._format_results()

    def run_relationship_inspection(self) -> Dict[str, Any]:
        """Inspect all relationships in detail."""
        print("\n" + "="*80)
        print("NEO4J RELATIONSHIP INSPECTION")
        print("="*80)

        # HAS_ELEMENT relationships (sample)
        result1 = self.execute_query(
            "HAS_ELEMENT Relationships (first 20)",
            """
            MATCH (p:Page)-[r:HAS_ELEMENT]->(e:Element)
            RETURN 
              p.page_id as PageId,
              e.element_id as ElementId,
              e.element_original_id as OriginalId
            LIMIT 20
            """
        )
        self.results.append(result1)

        # HAS_ELEMENT count by page
        result2 = self.execute_query(
            "HAS_ELEMENT Count by Page",
            """
            MATCH (p:Page)-[r:HAS_ELEMENT]->(e:Element)
            RETURN 
              p.page_id as PageId,
              COUNT(r) as ElementCount
            ORDER BY p.timestamp ASC
            """
        )
        self.results.append(result2)

        # LEADS_TO relationships
        result3 = self.execute_query(
            "All LEADS_TO Relationships",
            """
            MATCH (e:Element)-[r:LEADS_TO]->(p:Page)
            RETURN 
              e.element_id as ElementId,
              e.element_original_id as OriginalId,
              r.action_name as ActionName,
              r.confidence_score as Confidence,
              p.page_id as TargetPageId
            ORDER BY e.element_original_id ASC
            """
        )
        self.results.append(result3)

        # COMPOSED_OF relationships
        result4 = self.execute_query(
            "All COMPOSED_OF Relationships",
            """
            MATCH (a:Action)-[r:COMPOSED_OF]->(e:Element)
            RETURN 
              a.action_id as ActionId,
              a.action_name as ActionName,
              r.order as ExecutionOrder,
              r.atomic_action as AtomicAction,
              e.element_id as ElementId
            ORDER BY a.action_id, r.order ASC
            """
        )
        self.results.append(result4)

        return self._format_results()

    def run_path_inspection(self) -> Dict[str, Any]:
        """Inspect execution paths."""
        print("\n" + "="*80)
        print("NEO4J EXECUTION PATH INSPECTION")
        print("="*80)

        # Step-by-step trace
        result1 = self.execute_query(
            "Step-by-Step Execution Trace",
            """
            MATCH (p1:Page)-[:HAS_ELEMENT]->(e:Element)-[:LEADS_TO]->(p2:Page)
            RETURN 
              p1.page_id as SourcePage,
              e.element_id as ElementId,
              e.element_original_id as ElementOriginalId,
              e.action_type as ActionType,
              p2.page_id as TargetPage
            ORDER BY p1.timestamp ASC
            """
        )
        self.results.append(result1)

        # Start and end pages
        result2 = self.execute_query(
            "Start Pages (no incoming LEADS_TO)",
            """
            MATCH (p:Page)
            WHERE NOT EXISTS { ()-[:LEADS_TO]->(p) }
            RETURN 
              p.page_id as PageId,
              p.description as Description,
              p.timestamp as Timestamp
            """
        )
        self.results.append(result2)

        result3 = self.execute_query(
            "End Pages (no outgoing LEADS_TO)",
            """
            MATCH (p:Page)
            WHERE NOT EXISTS { (p)-[:LEADS_TO]->() }
            RETURN 
              p.page_id as PageId,
              p.description as Description,
              p.timestamp as Timestamp
            """
        )
        self.results.append(result3)

        return self._format_results()

    def run_validation_checks(self) -> Dict[str, Any]:
        """Run data quality validation checks."""
        print("\n" + "="*80)
        print("NEO4J DATA VALIDATION CHECKS")
        print("="*80)

        # Check for orphaned pages
        result1 = self.execute_query(
            "Orphaned Pages (no HAS_ELEMENT)",
            """
            MATCH (p:Page)
            WHERE NOT EXISTS { (p)-[:HAS_ELEMENT]->() }
            RETURN 
              p.page_id as PageId,
              p.raw_page_url as Screenshot
            """
        )
        self.results.append(result1)

        # Check for unconnected elements
        result2 = self.execute_query(
            "Unconnected Elements (no HAS_ELEMENT)",
            """
            MATCH (e:Element)
            WHERE NOT EXISTS { ()-[:HAS_ELEMENT]->(e) }
            RETURN 
              COUNT(e) as UnconnectedElements,
              COLLECT(e.element_id) as ElementIds
            """
        )
        self.results.append(result2)

        # Check for missing embeddings
        result3 = self.execute_query(
            "Elements Missing Visual Embeddings",
            """
            MATCH (e:Element)
            WHERE e.visual_embedding_id IS NULL
            RETURN 
              COUNT(e) as MissingEmbeddings,
              COLLECT(e.element_id) as ElementIds
            """
        )
        self.results.append(result3)

        # Check for duplicate elements
        result4 = self.execute_query(
            "Duplicate Elements (same original_id)",
            """
            MATCH (e1:Element), (e2:Element)
            WHERE e1.element_id < e2.element_id
              AND e1.element_original_id = e2.element_original_id
            RETURN 
              COUNT(*) as DuplicateCount,
              COLLECT(DISTINCT e1.element_original_id) as OriginalIds
            """
        )
        self.results.append(result4)

        # Check for duplicate pages
        result5 = self.execute_query(
            "Duplicate Pages (same screenshot)",
            """
            MATCH (p1:Page), (p2:Page)
            WHERE p1.page_id < p2.page_id
              AND p1.raw_page_url = p2.raw_page_url
            RETURN 
              COUNT(*) as DuplicateCount,
              COLLECT(DISTINCT p1.raw_page_url) as Screenshots
            """
        )
        self.results.append(result5)

        return self._format_results()

    def _format_results(self) -> Dict[str, Any]:
        """Format results for display."""
        return {
            "total_queries": len(self.results),
            "successful": sum(1 for r in self.results if r.status == "OK"),
            "empty": sum(1 for r in self.results if r.status == "EMPTY"),
            "errors": sum(1 for r in self.results if r.status == "ERROR")
        }

    def generate_report(self, output_file: str = "neo4j_inspection_report.json"):
        """Generate a JSON report of all query results."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "database": self.database,
            "summary": {
                "total_queries": len(self.results),
                "successful": sum(1 for r in self.results if r.status == "OK"),
                "empty": sum(1 for r in self.results if r.status == "EMPTY"),
                "errors": sum(1 for r in self.results if r.status == "ERROR"),
                "total_execution_time": sum(r.execution_time for r in self.results)
            },
            "queries": [
                {
                    "name": r.name,
                    "status": r.status,
                    "row_count": r.row_count,
                    "execution_time": r.execution_time,
                    "results": r.result,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Report saved to {output_file}")
        return report

    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        for result in self.results:
            status_icon = "✅" if result.status == "OK" else "⚠️" if result.status == "EMPTY" else "❌"
            print(f"{status_icon} {result.name}: {result.row_count} rows ({result.execution_time:.3f}s)")
            if result.error_message:
                print(f"   Error: {result.error_message}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j Database Inspector")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Database name")
    parser.add_argument("--output", default="neo4j_inspection_report.json", help="Output report file")

    args = parser.parse_args()

    inspector = Neo4jInspector(args.uri, args.user, args.password, args.database)

    try:
        print(f"\n🔍 Connecting to Neo4j at {args.uri}...")
        
        # Run all inspections
        inspector.run_health_check()
        inspector.run_node_inspection()
        inspector.run_relationship_inspection()
        inspector.run_path_inspection()
        inspector.run_validation_checks()

        # Print summary
        inspector.print_summary()

        # Generate report
        report = inspector.generate_report(args.output)
        
        # Print diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        stats = report["summary"]
        total_nodes = next(
            (r["results"][0] for r in report["queries"] if "Node Count" in r["name"]),
            None
        )
        
        if not total_nodes or all(v == 0 for v in total_nodes.values()):
            print("❌ DATABASE IS EMPTY")
            print("   Cause: history_steps was empty when json2db() ran")
            print("   Fix: Apply Patch 2 + 9 to populate history_steps")
        else:
            print("✅ DATABASE HAS DATA")
            print(f"   Pages: {total_nodes.get('Count', 0)}")
            print(f"   Elements: {total_nodes.get('Count', 0)}")
            print("   Status: Check report for details")

    finally:
        inspector.close()


if __name__ == "__main__":
    main()