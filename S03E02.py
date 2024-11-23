import os
import uuid
from loguru import logger
from utils_ai import aidevs_send_answer, openai_get_embedding
from utils_files_and_text import group_files_by_type
from utils_qdrant import qdrant_create_collection, qdrant_upsert, query_similar_text


def run():
    directory_weapons: str = "data/pliki_z_fabryki/weapons_tests/do-not-share"
    collection_name: str = "S03E02_large"
    question: str = os.getenv("S03E02_TASK_QUESTION")
    weapon_files = group_files_by_type(directory_weapons, file_types={".txt": "Text"})
    qdrant_create_collection(collection_name, size=3072)
    for file_name in weapon_files["Text"]:
        with open(
            os.path.join(directory_weapons, file_name), "r", encoding="utf-8"
        ) as file:
            content = file.read().strip()
            embedding = openai_get_embedding(content, model="text-embedding-3-large")
            unique_id = str(uuid.uuid4())
            date = os.path.splitext(file_name)[0]
            qdrant_upsert(
                collection_name=collection_name,
                unique_id=unique_id,
                embedding=embedding,
                payload={
                    "file_name": file_name,
                    "content": content,
                    "date": date.replace("_", "-"),
                },
            )
    query_results = query_similar_text(
        query_text=question,
        collection_name=collection_name,
        top_k=3,
        embedding_model="text-embedding-3-large",
    )
    first_result_date = query_results[0].payload["date"]
    answer = first_result_date
    logger.info(f"ANSWER: {answer}")

    response_task = aidevs_send_answer(
        task=os.getenv("S03E02_TASK_NAME"), answer=answer
    )
    if response_task.status_code == 200:
        logger.success(f"Request successful! Response:, {response_task.content}")
    else:
        logger.warning(
            f"Request failed with status code: {response_task.status_code}"
            f"Response content: {response_task.content}"
        )


if __name__ == "__main__":
    run()
