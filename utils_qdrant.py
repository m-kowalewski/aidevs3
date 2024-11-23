from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Any, Dict, List

from utils_ai import openai_get_embedding


client = QdrantClient(host="localhost", port=6333)


def qdrant_create_collection(
    collection_name: str, size: int = 1536, distance=Distance.COSINE
):
    existing_collections: List[str] = [
        col.name for col in client.get_collections().collections
    ]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        logger.debug(f"Successfully created collection: {collection_name}")
    else:
        logger.debug(f"Collection: {collection_name} already exists.")


def qdrant_upsert(
    collection_name: str,
    unique_id: str,
    embedding: List[float],
    payload: Dict[str, Any],
):
    try:
        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=unique_id, vector=embedding, payload=payload)],
        )
        logger.success(f"Successfully added {unique_id} to {collection_name}.")
    except Exception as e:
        logger.error(f"Error adding {unique_id} to {collection_name}.\n{e}")


def qdrant_search(collection_name: str, query_vector: List[float], top_k: int):
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )


def query_similar_text(
    query_text: str,
    collection_name: str,
    top_k: int = 5,
    embedding_model: str = "text-embedding-3-small",
):
    query_vector = openai_get_embedding(query_text, model=embedding_model)
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )
