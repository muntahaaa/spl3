### Find Start Pages (Pages with No Incoming LEADS_TO)

```cypher
MATCH (p:Page)
WHERE NOT EXISTS { ()-[:LEADS_TO]->(p) }
RETURN 
  p.page_id as PageId,
  p.description as Description,
  p.timestamp as Timestamp;
```
## To check high level action description 
```cypher
MATCH (a:Action)
WHERE coalesce(a.is_high_level, false) = true
RETURN a.action_id, a.name, a.description
```
## To check the count of reasoning with elements 
```cypher
MATCH (e:Element)
WHERE e.reasoning IS NOT NULL AND e.reasoning <> ""
RETURN count(e) AS elements_with_reasoning;
```
## To check overall execution path status 
```cypher
MATCH (start:Page)
WHERE NOT ()-[:LEADS_TO]->(start)
MATCH (end:Page)
WHERE NOT (end)-[:LEADS_TO]->()
MATCH paths = (start)-[:HAS_ELEMENT|LEADS_TO*]->(end)
RETURN
  DISTINCT paths as ExecutionPath,
  LENGTH(paths) as PathLen;
```
## To check triplets 
**Page-> (has element)Element -> (leads to)Page**
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

## Delete all 
```cypher
MATCH (n)
DETACH DELETE n;
```
