import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import config
from pinecone import Pinecone, ServerlessSpec


class NodeType(Enum):
    PAGE    = "page"
    ELEMENT = "element"


@dataclass
class VectorData:
    id:        str
    values:    List[float]
    metadata:  Dict[str, Any]
    node_type: NodeType


class VectorStore:
    """Thin wrapper around a Pinecone serverless index."""

    def __init__(
        self,
        api_key:    Optional[str] = None,
        index_name: Optional[str] = None,
        dimension:  int = 2048,
        batch_size: int = 100,
    ):
        resolved_api_key = api_key or config.PINECONE_API_KEY
        resolved_index_name = index_name or getattr(config, "PINECONE_INDEX_NAME", "vectordb")

        self.pc         = Pinecone(api_key=resolved_api_key)
        self.index_name = resolved_index_name
        self.dimension  = dimension
        self.batch_size = batch_size
        self._ensure_index()
        self._validate_index_dimension()
        self.index = self.pc.Index(self.index_name)

    # ── index management ──────────────────────

    def _index_exists(self, index_name: str) -> bool:
        """Support multiple Pinecone SDK variants for index existence checks."""
        try:
            has_index_fn = getattr(self.pc, "has_index", None)
            if callable(has_index_fn):
                return bool(has_index_fn(index_name))

            list_indexes_fn = getattr(self.pc, "list_indexes", None)
            if callable(list_indexes_fn):
                indexes = list_indexes_fn()

                names_fn = getattr(indexes, "names", None)
                if callable(names_fn):
                    return index_name in names_fn()

                if isinstance(indexes, dict):
                    if "indexes" in indexes and isinstance(indexes["indexes"], list):
                        names = [i.get("name") for i in indexes["indexes"] if isinstance(i, dict)]
                        return index_name in names
                    return index_name in indexes.keys()

                if isinstance(indexes, list):
                    names = []
                    for item in indexes:
                        if isinstance(item, str):
                            names.append(item)
                        elif isinstance(item, dict):
                            names.append(item.get("name"))
                        else:
                            names.append(getattr(item, "name", None))
                    return index_name in names

            # Last resort: treat successful describe as existence.
            self.pc.describe_index(index_name)
            return True
        except Exception:
            return False

    def _is_index_ready(self, index_name: str) -> bool:
        """Handle both dict-like and object-like index status payloads."""
        desc = self.pc.describe_index(index_name)
        status = getattr(desc, "status", None)

        if isinstance(status, dict):
            return bool(status.get("ready"))

        ready_attr = getattr(status, "ready", None)
        if ready_attr is not None:
            return bool(ready_attr)

        if isinstance(desc, dict):
            status_dict = desc.get("status", {})
            if isinstance(status_dict, dict):
                return bool(status_dict.get("ready"))

        return False

    def _ensure_index(self):
        if not self._index_exists(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self._is_index_ready(self.index_name):
                time.sleep(1)

    def _get_index_dimension(self, index_name: str) -> Optional[int]:
        """Read index dimension from multiple Pinecone response shapes."""
        desc = self.pc.describe_index(index_name)

        if isinstance(desc, dict):
            dim = desc.get("dimension")
            if isinstance(dim, int):
                return dim

        dim_attr = getattr(desc, "dimension", None)
        if isinstance(dim_attr, int):
            return dim_attr

        spec = getattr(desc, "spec", None)
        if isinstance(spec, dict):
            dim = spec.get("dimension")
            if isinstance(dim, int):
                return dim
        spec_dim = getattr(spec, "dimension", None)
        if isinstance(spec_dim, int):
            return spec_dim

        return None

    def _validate_index_dimension(self) -> None:
        """Fail fast if the configured embedding size doesn't match index schema."""
        actual_dim = self._get_index_dimension(self.index_name)
        if actual_dim is None:
            return
        if actual_dim != self.dimension:
            raise ValueError(
                f"Pinecone index '{self.index_name}' has dimension {actual_dim}, "
                f"but configured vector dimension is {self.dimension}. "
                "Recreate the index with the configured dimension or change the "
                "configured dimension and embedding model output to match."
            )

    # ── write ─────────────────────────────────

    def upsert_batch(self, vectors: List[VectorData]) -> bool:
        try:
            by_type: Dict[str, list] = {}
            for vec in vectors:
                ns = vec.node_type.value
                by_type.setdefault(ns, [])
                meta = {
                    k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                    for k, v in vec.metadata.items()
                }
                by_type[ns].append({"id": vec.id, "values": vec.values, "metadata": meta})

            for ns, vecs in by_type.items():
                total = len(vecs)
                if total == 0:
                    continue
                for i in range(0, total, self.batch_size):
                    self.index.upsert(vectors=vecs[i : i + self.batch_size], namespace=ns)
            return True
        except Exception as exc:
            print(f"[VectorStore] upsert_batch failed: {exc}")
            return False

    # ── read ──────────────────────────────────

    def query_similar(
        self,
        query_vector: List[float],
        node_type:    NodeType,
        top_k:        int = 5,
        filter_dict:  Optional[Dict] = None,
    ) -> Dict:
        try:
            results = self.index.query(
                namespace=node_type.value,
                vector=query_vector,
                top_k=top_k,
                include_values=True,
                include_metadata=True,
                filter=filter_dict,
            )
            for match in results.get("matches", []):
                for k, v in match.get("metadata", {}).items():
                    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                        try:
                            match["metadata"][k] = json.loads(v)
                        except json.JSONDecodeError:
                            pass
            return results
        except Exception as exc:
            print(f"[VectorStore] query_similar failed: {exc}")
            return {}

    # ── delete ────────────────────────────────

    def delete_vectors(self, ids: List[str], node_type: NodeType) -> bool:
        try:
            self.index.delete(ids=ids, namespace=node_type.value)
            return True
        except Exception as exc:
            print(f"[VectorStore] delete_vectors failed: {exc}")
            return False

    def get_stats(self) -> Dict:
        try:
            return self.index.describe_index_stats()
        except Exception as exc:
            print(f"[VectorStore] get_stats failed: {exc}")
            return {}
