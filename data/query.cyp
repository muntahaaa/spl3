// Queries extracted from data/graph_db.py
// Run these in Neo4j Browser. Set parameters before each query as needed.

// -----------------------------------------------------------------------------
// update_node_property(node_type provided)
// Python builds this dynamically with label and id field.
// Use one of the concrete variants below.
// -----------------------------------------------------------------------------

// Page variant
:param node_id => "page_001";
:param property_name => "title";
:param property_value => "Updated Page Title";

MATCH (n:Page)
WHERE n.page_id = $node_id
SET n[$property_name] = $property_value
RETURN n;

// Element variant
:param node_id => "element_001";
:param property_name => "element_type";
:param property_value => "button";

MATCH (n:Element)
WHERE n.element_id = $node_id
SET n[$property_name] = $property_value
RETURN n;

// Action variant
:param node_id => "action_001";
:param property_name => "status";
:param property_value => "active";

MATCH (n:Action)
WHERE n.action_id = $node_id
SET n[$property_name] = $property_value
RETURN n;

// -----------------------------------------------------------------------------
// update_node_property(node_type not provided)
// -----------------------------------------------------------------------------

:param node_id => "any_001";
:param property_name => "note";
:param property_value => "updated via fallback query";

MATCH (n)
WHERE n.page_id = $node_id OR n.element_id = $node_id OR n.action_id = $node_id
SET n[$property_name] = $property_value
RETURN n;

// -----------------------------------------------------------------------------
// create_node(label, properties)
// Python uses a dynamic label. Cypher cannot parameterize labels directly.
// Use one of these concrete runnable label-specific variants.
// -----------------------------------------------------------------------------

// Create Page node
:param properties => {
	page_id: "page_001",
	timestamp: 1713254400,
	visual_embedding_id: "emb_001",
	other_info: "{\"source\":\"manual\"}"
};

CREATE (n:Page $properties)
RETURN elementId(n) AS node_id;

// Create Element node
:param properties => {
	element_id: "element_001",
	element_type: "button",
	visual_embedding_id: "emb_elm_001",
	other_info: "{\"color\":\"blue\"}"
};

CREATE (n:Element $properties)
RETURN elementId(n) AS node_id;

// Create Action node
:param properties => {
	action_id: "action_001",
	action_name: "click"
};

CREATE (n:Action $properties)
RETURN elementId(n) AS node_id;

// -----------------------------------------------------------------------------
// add_element_to_page(page_id, element_id)
// -----------------------------------------------------------------------------

:param page_id => "page_001";
:param element_id => "element_001";

MATCH (p:Page {page_id: $page_id})
MATCH (e:Element {element_id: $element_id})
MERGE (p)-[r:HAS_ELEMENT]->(e)
RETURN type(r) AS rel_type;

// -----------------------------------------------------------------------------
// add_element_to_action(action_id, element_id, order, atomic_action, action_params)
// -----------------------------------------------------------------------------

:param action_id => "action_001";
:param element_id => "element_001";
:param order => 1;
:param atomic_action => "click";
:param action_params => "{\"x\":120,\"y\":320}";

MATCH (a:Action {action_id: $action_id})
MATCH (e:Element {element_id: $element_id})
MERGE (a)-[r:COMPOSED_OF {
	order: $order,
	atomic_action: $atomic_action,
	action_params: $action_params
}]->(e)
RETURN type(r) AS rel_type;

// -----------------------------------------------------------------------------
// get_page_elements(page_id)
// -----------------------------------------------------------------------------

:param page_id => "page_001";

MATCH (p:Page {page_id: $page_id})-[:HAS_ELEMENT]->(e:Element)
RETURN e;

// -----------------------------------------------------------------------------
// get_action_sequence(action_id)
// -----------------------------------------------------------------------------

:param action_id => "action_001";

MATCH (a:Action {action_id: $action_id})-[r:COMPOSED_OF]->(e:Element)
RETURN e.element_id AS element_id,
			 e.element_type AS element_type,
			 r.order AS order,
			 r.atomic_action AS atomic_action,
			 r.action_params AS action_params
ORDER BY r.order;

// -----------------------------------------------------------------------------
// add_element_leads_to(element_id, target_id, action_name, action_params, confidence_score)
// -----------------------------------------------------------------------------

:param element_id => "element_001";
:param target_id => "page_002";
:param action_name => "tap";
:param action_params => "{\"pressure\":0.8}";
:param confidence_score => 0.95;

MATCH (e:Element {element_id: $element_id})
MATCH (t:Page {page_id: $target_id})
MERGE (e)-[r:LEADS_TO {
	action_name: $action_name,
	action_params: $action_params,
	confidence_score: $confidence_score
}]->(t)
RETURN type(r) AS rel_type;

// -----------------------------------------------------------------------------
// get_chain_start_nodes()
// -----------------------------------------------------------------------------

MATCH (n:Page)
WHERE NOT EXISTS { MATCH ()-[:LEADS_TO]->(n) }
RETURN n;

// -----------------------------------------------------------------------------
// get_chain_from_start(start_page_id)
// -----------------------------------------------------------------------------

:param start_page_id => "page_001";

MATCH path = (start:Page {page_id: $start_page_id})-[:HAS_ELEMENT|LEADS_TO*]->(end:Page)
WHERE NOT EXISTS { (end)-[:HAS_ELEMENT]->() }
WITH path, relationships(path) AS rels, nodes(path) AS nodes
WITH DISTINCT [n IN nodes | n{.*}] AS node_props,
							[r IN rels | r{.*}] AS rel_props
RETURN node_props, rel_props;
