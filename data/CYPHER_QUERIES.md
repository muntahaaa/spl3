# Neo4j Cypher Queries: Complete Database Inspection

This file contains Cypher queries to inspect all nodes (Pages, Elements, Actions) and relationships in your Neo4j graph database.

---

## Part 1: Node Count & Overview Queries

### 1.1 Count All Nodes by Type

```cypher
MATCH (n)
RETURN 
  labels(n)[0] as NodeType,
  count(n) as Count
ORDER BY Count DESC;
```

**Expected output (if working)**:
```
NodeType      Count
Page          3
Element       44
Action        3
```

**What to do if empty**: If all counts are 0, your `history_steps` was empty when `json2db()` ran.

---

### 1.2 Total Node Statistics

```cypher
MATCH (n)
RETURN 
  COUNT(n) as TotalNodes,
  COUNT(DISTINCT labels(n)[0]) as NodeTypes;
```

**Expected**: 50 total nodes, 3 types

---

### 1.3 List All Unique Node Types

```cypher
CALL db.labels() 
YIELD label 
RETURN label 
ORDER BY label;
```

**Expected output**:
```
Action
Element
Page
```

---

## Part 2: Page Node Queries

### 2.1 Count All Page Nodes

```cypher
MATCH (p:Page)
RETURN 
  COUNT(p) as TotalPages,
  COUNT(DISTINCT p.page_id) as UniquePageIds;
```

**Expected**: 3+ pages (one per step + final page)

---

### 2.2 List All Pages with Details

```cypher
MATCH (p:Page)
RETURN 
  p.page_id as PageId,
  p.description as Description,
  p.raw_page_url as Screenshot,
  p.timestamp as Timestamp,
  p.other_info as OtherInfo
ORDER BY p.timestamp ASC;
```

**Expected output**:
```
PageId          Description  Screenshot              Timestamp    OtherInfo
page_uuid_1     ...          ./screenshots/step0.png  1700000000   {"step": 0, "task_info": {...}}
page_uuid_2     ...          ./screenshots/step1.png  1700000005   {"step": 1}
page_uuid_3     ...          ./screenshots/step2.png  1700000010   {"step": 2}
```

---

### 2.3 Get Page with All Its Elements

```cypher
MATCH (p:Page)
OPTIONAL MATCH (p)-[:HAS_ELEMENT]->(e:Element)
WITH 
  p, 
  COUNT(e) AS ElementCount, 
  COLLECT(e.element_id) AS ElementIds
RETURN 
  p.page_id AS PageId,
  p.description AS PageDescription,
  ElementCount,
  ElementIds
ORDER BY p.timestamp ASC;              
```

**Expected**:
```
PageId          PageDescription  ElementCount  ElementIds
page_uuid_1     ...              13            [elem_uuid_1, elem_uuid_2, ...]
page_uuid_2     ...              17            [elem_uuid_14, elem_uuid_15, ...]
page_uuid_3     ...              14            [elem_uuid_31, elem_uuid_32, ...]
```

---

### 2.4 Find Pages with No Elements (Empty Pages)

```cypher
MATCH (p:Page)
WHERE NOT EXISTS { (p)-[:HAS_ELEMENT]->() }
RETURN 
  p.page_id as PageId,
  p.raw_page_url as Screenshot,
  p.timestamp as Timestamp;
```

**Expected**: Should be empty (all pages should have elements)

**If not empty**: These pages were created but their elements were never linked.

---

### 2.5 Find Start Pages (Pages with No Incoming LEADS_TO)

```cypher
MATCH (p:Page)
WHERE NOT EXISTS { ()-[:LEADS_TO]->(p) }
RETURN 
  p.page_id as PageId,
  p.description as Description,
  p.timestamp as Timestamp;
```

**Expected**: Should return 1 page (the first page/homepage)

---

### 2.6 Find End Pages (Pages with No Outgoing LEADS_TO)

```cypher
MATCH (p:Page)
WHERE NOT EXISTS { (p)-[:LEADS_TO]->() }
RETURN 
  p.page_id as PageId,
  p.description as Description,
  p.timestamp as Timestamp;
```

**Expected**: Should return 1 page (the final/completion page)

---

## Part 3: Element Node Queries

### 3.1 Count All Element Nodes

```cypher
MATCH (e:Element)
RETURN 
  COUNT(e) as TotalElements,
  COUNT(DISTINCT e.element_id) as UniqueElementIds,
  COUNT(DISTINCT e.element_original_id) as UniqueOriginalIds;
```

**Expected**: 44+ elements (13 from step 1 + 17 from step 2 + 14 from step 3)

---

### 3.2 List All Elements with Details

```cypher
MATCH (e:Element)
RETURN 
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.action_type as ActionType,
  e.bounding_box as BBox,
  e.description as Description,
  e.visual_embedding_id as EmbeddingId
ORDER BY e.element_original_id ASC
LIMIT 20;
```

**Expected output** (first 20):
```
ElementId          OriginalId  ActionType  BBox                          Description  EmbeddingId
elem_uuid_1        0           tap         [0.11, 0.16, 0.79, 0.18]      ...          pin_elem_uuid_1
elem_uuid_2        1           tap         [0.32, 0.74, 0.44, 0.76]      ...          pin_elem_uuid_2
...
```

---

### 3.3 Count Elements by Action Type

```cypher
MATCH (e:Element)
RETURN 
  e.action_type as ActionType,
  COUNT(e) as Count
ORDER BY Count DESC;
```

**Expected output**:
```
ActionType  Count
tap         40
text        3
swipe       1
```

---

### 3.4 Find Elements with Visual Embeddings

```cypher
MATCH (e:Element)
WHERE e.visual_embedding_id IS NOT NULL
RETURN 
  COUNT(e) as ElementsWithEmbeddings,
  COUNT(DISTINCT e.visual_embedding_id) as UniqueEmbeddings;
```

**Expected**: 44+ elements with embeddings (ready for Pinecone search)

---

### 3.5 Find Elements Missing Visual Embeddings

```cypher
MATCH (e:Element)
WHERE e.visual_embedding_id IS NULL
RETURN 
  COUNT(e) as ElementsMissingEmbeddings,
  COLLECT(e.element_id) as MissingElementIds;
```

**Expected**: Should be empty (all should have embeddings)

**If not empty**: These elements were created but not vectorized.

---

### 3.6 Find Element with Specific Content

```cypher
MATCH (e:Element)
WHERE e.other_info CONTAINS "search"
RETURN 
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.other_info as OtherInfo;
```

**Expected**: Elements with content containing "search"

---

### 3.7 Get Elements by Page

```cypher
MATCH (p:Page)-[:HAS_ELEMENT]->(e:Element)
RETURN 
  p.page_id as PageId,
  p.timestamp as PageTimestamp,
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.action_type as ActionType,
  e.other_info as Content
ORDER BY p.timestamp ASC, e.element_original_id ASC;
```

**Expected**: All 44 elements grouped by their page

---

## Part 4: Relationship Queries

### 4.1 Count All Relationships by Type

```cypher
MATCH ()-[r]->()
RETURN 
  type(r) as RelationType,
  COUNT(r) as Count
ORDER BY Count DESC;
```

**Expected output**:
```
RelationType  Count
HAS_ELEMENT   44 (13 from page1 + 17 from page2 + 14 from page3)
LEADS_TO      2  (page1→page2, page2→page3)
COMPOSED_OF   0  (no action shortcuts created yet)
```

---

### 4.2 List All HAS_ELEMENT Relationships

```cypher
MATCH (p:Page)-[r:HAS_ELEMENT]->(e:Element)
RETURN 
  p.page_id as PageId,
  p.timestamp as PageTimestamp,
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.action_type as ActionType
ORDER BY p.timestamp ASC, e.element_original_id ASC
LIMIT 30;
```

**Expected**: First 30 of the 44 page-element relationships

---

### 4.3 Count HAS_ELEMENT Relationships by Page

```cypher
MATCH (p:Page)-[r:HAS_ELEMENT]->(e:Element)
RETURN 
  p.page_id as PageId,
  p.timestamp as PageTimestamp,
  COUNT(r) as ElementCount
ORDER BY p.timestamp ASC;
```

**Expected**:
```
PageId          PageTimestamp  ElementCount
page_uuid_1     1700000000     13
page_uuid_2     1700000005     17
page_uuid_3     1700000010     14
```

---

### 4.4 Visualize Page → Element Hierarchy (Tree Structure)

```cypher
MATCH (p:Page)-[r:HAS_ELEMENT]->(e:Element)
WITH p, COLLECT({element_id: e.element_id, original_id: e.element_original_id}) as elements
RETURN 
  p.page_id as PageId,
  p.timestamp as Timestamp,
  SIZE(elements) as ElementCount,
  elements
ORDER BY p.timestamp ASC;
```

**Expected**: Shows tree structure of pages and their elements

---

### 4.5 List All LEADS_TO Relationships

```cypher
MATCH (e:Element)-[r:LEADS_TO]->(p:Page)
RETURN 
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.action_type as ActionType,
  r.action_name as ActionName,
  r.confidence_score as Confidence,
  p.page_id as TargetPageId,
  p.timestamp as TargetPageTimestamp
ORDER BY e.element_original_id ASC;
```

**Expected output**:
```
ElementId       OriginalId  ActionType  ActionName  Confidence  TargetPageId    TargetPageTimestamp
elem_uuid_1     1           tap         tap         0.95        page_uuid_2     1700000005
elem_uuid_30    3           tap         tap         0.95        page_uuid_3     1700000010
```

**Expected count**: 2 relationships (one for transition from page1→page2, one from page2→page3)

---

### 4.6 Count LEADS_TO Relationships

```cypher
MATCH ()-[r:LEADS_TO]->()
RETURN 
  COUNT(r) as TotalLeadsTo,
  COUNT(DISTINCT r.action_name) as UniqueActionNames;
```

**Expected**: 2 total relationships, 1 unique action name

---

### 4.7 Find Elements with No Outgoing LEADS_TO (Dead Ends)

```cypher
MATCH (e:Element)
WHERE NOT EXISTS { (e)-[:LEADS_TO]->() }
RETURN 
  COUNT(e) as DeadEndElements,
  COLLECT(e.element_id) as ElementIds;
```

**Expected**: 42 elements (all except the 2 that trigger page transitions)

---

### 4.8 Find Pages with No Incoming LEADS_TO (Orphan Pages)

```cypher
MATCH (p:Page)
WHERE NOT EXISTS { ()-[:LEADS_TO]->(p) }
RETURN 
  COUNT(p) as OrphanPages,
  COLLECT(p.page_id) as PageIds;
```

**Expected**: 1 page (the first page)

---

## Part 5: Action Node Queries

### 5.1 Count All Action Nodes

```cypher
MATCH (a:Action)
RETURN 
  COUNT(a) as TotalActions,
  COUNT(DISTINCT a.action_id) as UniqueActionIds;
```

**Expected**: 0 (unless you applied the patches to create action nodes)

**If 0**: This is expected - action node creation is not implemented yet (Patch 6-7)

---

### 5.2 List All Actions with Details

```cypher
MATCH (a:Action)
RETURN 
  a.action_id as ActionId,
  a.action_name as ActionName,
  a.timestamp as Timestamp,
  a.step as Step,
  a.action_result as Result
ORDER BY a.step ASC;
```

**Expected**: Empty result (feature not yet implemented)

---

### 5.3 Get Action Composition (COMPOSED_OF)

```cypher
MATCH (a:Action)-[r:COMPOSED_OF]->(e:Element)
RETURN 
  a.action_id as ActionId,
  a.action_name as ActionName,
  r.order as ExecutionOrder,
  r.atomic_action as AtomicAction,
  e.element_id as ElementId,
  e.element_original_id as OriginalId
ORDER BY a.action_id, r.order ASC;
```

**Expected**: Empty result (feature not yet implemented)

---

## Part 6: Complex Path Queries

### 6.1 Get Complete Task Execution Path (Page → Element → Page)

```cypher
MATCH path = (start:Page)-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
WHERE NOT ()-[:LEADS_TO]->(start) AND NOT (end)-[:LEADS_TO]->()
RETURN 
  path as ExecutionPath,
  LENGTH(path) as PathLength
LIMIT 5;
```

**Expected**: The complete task execution flow from first to last page

---

### 6.2 Trace Step-by-Step Execution

```cypher
MATCH (p1:Page)-[:HAS_ELEMENT]->(e:Element)-[:LEADS_TO]->(p2:Page)
RETURN 
  p1.page_id as SourcePage,
  p1.timestamp as SourceTimestamp,
  e.element_id as ElementId,
  e.element_original_id as ElementOriginalId,
  e.action_type as ActionType,
  p2.page_id as TargetPage,
  p2.timestamp as TargetTimestamp
ORDER BY p1.timestamp ASC;
```

**Expected**: Shows each step-by-step transition in the task

---

### 6.3 Get Full Execution Sequence

```cypher
MATCH (start:Page)
WHERE NOT ()-[:LEADS_TO]->(start)
MATCH path = (start)-[:HAS_ELEMENT|LEADS_TO*]->()
WITH DISTINCT path
UNWIND nodes(path) as node
RETURN 
  CASE WHEN node:Page THEN 'Page' ELSE 'Element' END as NodeType,
  COALESCE(node.page_id, node.element_id) as NodeId,
  COALESCE(node.timestamp, '') as Timestamp,
  COALESCE(node.element_original_id, '') as OriginalId
ORDER BY Timestamp ASC;
```

**Expected**: Complete linearized execution sequence

---

### 6.4 Find All Possible Paths from Start to End

```cypher
MATCH (start:Page)
WHERE NOT ()-[:LEADS_TO]->(start)
MATCH (end:Page)
WHERE NOT (end)-[:LEADS_TO]->()
MATCH paths = (start)-[:HAS_ELEMENT|LEADS_TO*]->(end)
RETURN 
  DISTINCT paths as ExecutionPath,
  LENGTH(paths) as PathLength
LIMIT 10;
```

**Expected**: All possible execution paths (should be 1 for simple linear task)

---

## Part 7: Data Quality & Validation Queries

### 7.1 Validate Page Node Completeness

```cypher
MATCH (p:Page)
RETURN 
  p.page_id as PageId,
  CASE WHEN p.page_id IS NULL THEN 'MISSING' ELSE 'OK' END as page_id,
  CASE WHEN p.timestamp IS NULL THEN 'MISSING' ELSE 'OK' END as timestamp,
  CASE WHEN p.raw_page_url IS NULL THEN 'MISSING' ELSE 'OK' END as raw_page_url,
  CASE WHEN p.other_info IS NULL THEN 'MISSING' ELSE 'OK' END as other_info;
```

**Expected**: All "OK" for all pages

---

### 7.2 Validate Element Node Completeness

```cypher
MATCH (e:Element)
RETURN 
  e.element_id as ElementId,
  CASE WHEN e.element_id IS NULL THEN 'MISSING' ELSE 'OK' END as element_id,
  CASE WHEN e.element_original_id IS NULL THEN 'MISSING' ELSE 'OK' END as original_id,
  CASE WHEN e.action_type IS NULL THEN 'MISSING' ELSE 'OK' END as action_type,
  CASE WHEN e.visual_embedding_id IS NULL THEN 'MISSING' ELSE 'OK' END as embedding_id
LIMIT 20;
```

**Expected**: All "OK" for all elements

---

### 7.3 Find Duplicate Pages

```cypher
MATCH (p1:Page), (p2:Page)
WHERE p1.page_id <> p2.page_id
  AND p1.raw_page_url = p2.raw_page_url
RETURN 
  p1.page_id as Page1,
  p2.page_id as Page2,
  p1.raw_page_url as SharedScreenshot
LIMIT 10;
```

**Expected**: Empty (no duplicate pages)

**If not empty**: Duplicate pages from the same screenshot

---

### 7.4 Find Duplicate Elements

```cypher
MATCH (e1:Element), (e2:Element)
WHERE e1.element_id <> e2.element_id
  AND e1.element_original_id = e2.element_original_id
RETURN 
  e1.element_id as Element1,
  e2.element_id as Element2,
  e1.element_original_id as SharedOriginalId
LIMIT 10;
```

**Expected**: Empty (no duplicate elements)

---

### 7.5 Check Visual Embedding References

```cypher
MATCH (e:Element)
RETURN 
  COUNT(e) as TotalElements,
  COUNT(CASE WHEN e.visual_embedding_id IS NOT NULL THEN 1 END) as WithEmbeddings,
  COUNT(CASE WHEN e.visual_embedding_id IS NULL THEN 1 END) as WithoutEmbeddings;
```

**Expected**:
```
TotalElements  WithEmbeddings  WithoutEmbeddings
44             44              0
```

---

## Part 8: Statistics & Summary Queries

### 8.1 Database Overview

```cypher
MATCH (n)
WITH 
  COUNT(n) as TotalNodes,
  COUNT(DISTINCT labels(n)[0]) as NodeTypes
MATCH ()-[r]->()
WITH TotalNodes, NodeTypes, COUNT(r) as TotalRelationships
RETURN 
  TotalNodes,
  NodeTypes,
  TotalRelationships;
```

**Expected**:
```
TotalNodes  NodeTypes  TotalRelationships
50          3          46 (44 HAS_ELEMENT + 2 LEADS_TO)
```

---

### 8.2 Detailed Statistics

```cypher
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
  ComposedOfCount;
```

**Expected**:
```
PageCount  ElementCount  ActionCount  HasElementCount  LeadsToCount  ComposedOfCount
3          44            0            44               2             0
```

---

### 8.3 Task Execution Timeline

```cypher
MATCH (p:Page)
RETURN 
  p.timestamp as EventTime,
  'PAGE_START' as EventType,
  p.page_id as PageId,
  COUNT((p)-[:HAS_ELEMENT]->()) as RelatedElements
ORDER BY EventTime ASC;
```

**Expected**: Shows timeline of page creation

---

## Part 9: Search & Retrieval Queries

### 9.1 Find Element by Original ID

```cypher
MATCH (e:Element {element_original_id: 0})
RETURN 
  e.element_id as ElementId,
  e.element_original_id as OriginalId,
  e.action_type as ActionType,
  e.bounding_box as BBox,
  e.other_info as Content;
```

**Expected**: The first element from ss1.json

---

### 9.2 Find Page by Timestamp

```cypher
MATCH (p:Page)
WHERE p.timestamp > 1700000000 AND p.timestamp < 1700001000
RETURN 
  p.page_id as PageId,
  p.timestamp as Timestamp,
  p.raw_page_url as Screenshot;
```

**Expected**: Pages created within timestamp range

---

### 9.3 Find All Elements of a Specific Type (e.g., "tap")

```cypher
MATCH (e:Element {action_type: "tap"})
RETURN 
  COUNT(e) as TapElements,
  COLLECT(DISTINCT e.element_original_id) as ElementIds;
```

**Expected**: ~40 tap elements

---

### 9.4 Find Elements with Specific Bounding Box Range

```cypher
MATCH (e:Element)
WHERE e.bounding_box[0] > 0.1 AND e.bounding_box[2] < 0.9
RETURN 
  COUNT(e) as ElementsInRange,
  COLLECT(e.element_id) as ElementIds;
```

**Expected**: Elements with x-coordinate between 0.1 and 0.9

---

## Part 10: Diagnostic Queries

### 10.1 Check if Database Is Empty

```cypher
RETURN 
  EXISTS { MATCH (n) RETURN n } as HasNodes,
  EXISTS { MATCH ()-[r]->() RETURN r } as HasRelationships;
```

**Expected**:
```
HasNodes  HasRelationships
true      true
```

**If both false**: Database is empty - `json2db()` was never run or `history_steps` was empty.

---

### 10.2 Database Health Check

```cypher
RETURN 
  apoc.db.info().storeSize as DatabaseSize,
  apoc.db.info().transactionQueueSize as PendingTransactions;
```

**Note**: Requires APOC plugin. Alternative without APOC:

```cypher
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
  END as DatabaseStatus;
```

**Expected**: HEALTHY (if data was stored)

---

### 10.3 Find Unconnected Components

```cypher
MATCH (n)
WHERE NOT EXISTS { (n)-[]->() } AND NOT EXISTS { ()-[]->(n) }
RETURN 
  COUNT(n) as IsolatedNodes,
  COLLECT(DISTINCT labels(n)[0]) as IsolatedNodeTypes;
```

**Expected**: Empty (all nodes should be connected)

**If not empty**: These nodes exist but aren't linked to the main task graph.

---

## Part 11: Execution & Debugging

### 11.1 Run Basic Health Check

Execute this query first to diagnose the status:

```cypher
MATCH (p:Page)
RETURN COUNT(p) as Pages

UNION ALL

MATCH (e:Element)
RETURN COUNT(e) as Elements

UNION ALL

MATCH ()-[r:HAS_ELEMENT]->()
RETURN COUNT(r) as HAS_ELEMENT_Rels

UNION ALL

MATCH ()-[r:LEADS_TO]->()
RETURN COUNT(r) as LEADS_TO_Rels;
```

**If you get zeros**: Your `history_steps` was empty when `json2db()` ran. Apply Patches 1-2 and 9.

---

### 11.2 Validate Complete Data Flow

```cypher
// Check Pages
MATCH (p:Page)
WITH COUNT(p) as Pages
// Check Elements
MATCH (e:Element)
WITH Pages, COUNT(e) as Elements
// Check HAS_ELEMENT relationships
MATCH ()-[r:HAS_ELEMENT]->()
WITH Pages, Elements, COUNT(r) as HasElement
// Check LEADS_TO relationships
MATCH ()-[r:LEADS_TO]->()
WITH Pages, Elements, HasElement, COUNT(r) as LeadsTo
// Validate
RETURN 
  Pages,
  Elements,
  HasElement,
  LeadsTo,
  CASE 
    WHEN Pages = 0 THEN 'EMPTY - Run patches to populate history_steps'
    WHEN Pages > 0 AND HasElement = 0 THEN 'ERROR - Pages exist but no elements'
    WHEN Pages > 0 AND Elements > 0 AND HasElement = Pages THEN 'OK - Structure valid'
    ELSE 'PARTIAL - Some relationships missing'
  END as Status;
```

---

## Usage Instructions

1. **Copy any query from this file**
2. **Paste into Neo4j Browser** (http://localhost:7687)
3. **Press Ctrl+Enter to execute**
4. **Review the results**

### Common Issues & Fixes

| Query Result | Interpretation | Fix |
|---|---|---|
| All counts = 0 | Database empty | Apply Patch 2 + 9 (populate `history_steps`) |
| Pages > 0, HAS_ELEMENT = 0 | Pages created but elements missing | Check element creation logic in `json2db()` |
| Elements > 0, LEADS_TO = 0 | Elements exist but no transitions | Check `add_element_leads_to()` calls |
| Orphaned nodes found | Some nodes not connected | Check relationship creation |
| Duplicate elements | Data integrity issue | Check for re-running `json2db()` multiple times |
