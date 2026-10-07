"""Delete one stored case while protecting shared recordings; retry vector cleanup."""
import json
import uuid
from pathlib import Path


def _decoded(value):
    if isinstance(value,str):
        try: return json.loads(value)
        except ValueError: return []
    return value or []


def _references(action):
    pages,elements=set(),set()
    for step in _decoded(action.get("element_sequence")):
        if not isinstance(step,dict): continue
        if step.get("element_id"): elements.add(step["element_id"])
        for key in ("source","destination"):
            page=step.get(key) or {}
            if isinstance(page,dict) and page.get("page_id"): pages.add(page["page_id"])
    final=_decoded(action.get("final_screen"))
    if isinstance(final,dict) and final.get("page_id"): pages.add(final["page_id"])
    return pages,elements


def _cleanup_vectors(vector,result):
    from data.vector_db import NodeType
    failed=[]
    for key,kind in (("pages",NodeType.PAGE),("elements",NodeType.ELEMENT),("actions",NodeType.ACTION)):
        ids=result.get(key,[])
        if not ids: continue
        try:
            if vector.delete_vectors(ids,kind) is not True: failed.append(kind.value)
        except Exception:
            failed.append(kind.value)
    result["vector_cleanup_failed"]=failed
    result["status"]="incomplete" if failed else "deleted"
    return result


def delete_case(db,vector,case_id):
    if not isinstance(case_id,str) or not case_id.strip():
        raise ValueError("Select a stored test case")
    def transaction(tx):
        record=tx.run("MATCH (a:Action {action_id:$id}) WHERE a.is_high_level=true RETURN a",id=case_id).single()
        if not record: raise ValueError("Stored test case no longer exists")
        own=dict(record["a"]);pages,element_ids=_references(own)
        protected_pages,protected_elements=set(),set()
        for row in tx.run("MATCH (a:Action) WHERE a.is_high_level=true AND a.action_id <> $id RETURN a",id=case_id):
            p,e=_references(dict(row["a"]));protected_pages.update(p);protected_elements.update(e)
        for row in tx.run("""MATCH (a:Action {action_id:$id})-[:COMPOSED_OF]->(e:Element)
            OPTIONAL MATCH (p:Page)-[:HAS_ELEMENT]->(e)
            OPTIONAL MATCH (e)-[:LEADS_TO]->(d:Page)
            RETURN e.element_id AS element,p.page_id AS source,d.page_id AS destination""",id=case_id):
            if row.get("element"): element_ids.add(row["element"])
            for key in ("source","destination"):
                if row.get(key): pages.add(row[key])
        # Relationships also protect legacy cases whose sequence has no page IDs.
        for row in tx.run("""MATCH (a:Action)-[:COMPOSED_OF]->(e:Element)
            WHERE a.is_high_level=true AND a.action_id <> $id
            OPTIONAL MATCH (p:Page)-[:HAS_ELEMENT]->(e)
            OPTIONAL MATCH (e)-[:LEADS_TO]->(d:Page)
            RETURN e.element_id AS element,p.page_id AS source,d.page_id AS destination""",id=case_id):
            if row.get("element"): protected_elements.add(row["element"])
            for key in ("source","destination"):
                if row.get(key): protected_pages.add(row[key])
        removable_pages=pages-protected_pages
        page_elements=set()
        for row in tx.run("""MATCH (p:Page) WHERE p.page_id IN $ids
            OPTIONAL MATCH (p)-[:HAS_ELEMENT]->(e:Element)
            RETURN p,collect(e.element_id) AS elements""",ids=sorted(removable_pages)):
            page_elements.update(row.get("elements") or [])
        removable_elements=(element_ids|page_elements)-protected_elements
        # An element attached to any retained page must remain available.
        shared=set()
        for row in tx.run("""MATCH (p:Page)-[:HAS_ELEMENT]->(e:Element)
            WHERE e.element_id IN $elements AND NOT p.page_id IN $pages
            RETURN DISTINCT e.element_id AS element""",elements=sorted(removable_elements),pages=sorted(removable_pages)):
            if row.get("element"): shared.add(row["element"])
        removable_elements-=shared
        low_actions=[]
        for row in tx.run("""MATCH (a:Action)-[:COMPOSED_OF]->(e:Element)
            WHERE coalesce(a.is_high_level,false)=false
            WITH a,collect(e.element_id) AS ids
            WHERE size(ids)>0 AND all(id IN ids WHERE id IN $elements)
            RETURN a.action_id AS action""",elements=sorted(removable_elements)):
            if row.get("action"): low_actions.append(row["action"])
        result={"case_id":case_id,"pages":sorted(removable_pages),"elements":sorted(removable_elements),"actions":sorted(set(low_actions+[case_id]))}
        # Write retry intent before graph deletion. Graph transaction is atomic.
        folder=Path("log/case_deletions");folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/(uuid.uuid4().hex+".json")
        result.update(cleanup_manifest=str(manifest.resolve()),graph_deleted=False)
        manifest.write_text(json.dumps(result,indent=2),encoding="utf-8")
        tx.run("MATCH (e:Element) WHERE e.element_id IN $ids DETACH DELETE e",ids=result["elements"]).consume()
        tx.run("MATCH (p:Page) WHERE p.page_id IN $ids DETACH DELETE p",ids=result["pages"]).consume()
        tx.run("MATCH (a:Action) WHERE a.action_id IN $ids DETACH DELETE a",ids=result["actions"]).consume()
        return result
    with db.driver.session(database=db.database) as session:
        result=session.execute_write(transaction)
    result["graph_deleted"]=True
    manifest=Path(result["cleanup_manifest"])
    manifest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    _cleanup_vectors(vector,result)
    manifest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def retry_case_cleanup(vector,manifest):
    root=Path("log/case_deletions").resolve()
    path=Path(manifest).resolve()
    if path.parent != root or path.suffix.lower() != ".json":
        raise ValueError("Invalid case cleanup manifest")
    result=json.loads(path.read_text(encoding="utf-8"))
    if result.get("graph_deleted") is not True:
        raise ValueError("Graph deletion was not confirmed; vector cleanup cannot be retried")
    _cleanup_vectors(vector,result)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
