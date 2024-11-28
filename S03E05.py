import json
import os
from loguru import logger
from typing import Dict, List, Tuple

from utils_ai import aidevs_send_answer, aidevs_s03e03_query
from utils_neo4j import (
    create_driver,
    close_connection,
    add_person,
    add_relationship,
    find_shortest_path,
)


def _fetch_data() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    query_all_users: str = "SELECT * FROM users;"
    response_all_users = aidevs_s03e03_query(query_all_users)
    parsed_data_all_users: List[Dict[str, str]] = json.loads(
        response_all_users.content.decode("utf-8")
    )["reply"]
    query_all_connections: str = "SELECT * FROM connections;"
    response_all_connections = aidevs_s03e03_query(query_all_connections)
    parsed_data_all_connections: List[Dict[str, str]] = json.loads(
        response_all_connections.content.decode("utf-8")
    )["reply"]
    return parsed_data_all_users, parsed_data_all_connections


def _insert_data(
    session, people: List[Dict[str, str]], connections: List[Tuple[str, str]]
) -> None:
    for person in people:
        session.write_transaction(add_person, person["id"], person["name"])
    for id1, id2 in connections:
        session.write_transaction(add_relationship, id1, id2)


def run():
    data_all_users, data_all_connections = _fetch_data()
    people: List[Dict[str, str]] = [
        {"id": item["id"], "name": item["username"]} for item in data_all_users
    ]
    connections: List[Tuple[str, str]] = [
        (item["user1_id"], item["user2_id"]) for item in data_all_connections
    ]
    start_name: str = "Rafał"
    end_name: str = "Barbara"

    driver = create_driver()
    try:
        with driver.session() as session:
            _insert_data(session, people, connections)
            shortest_path: List = find_shortest_path(session, start_name, end_name)
    finally:
        close_connection(driver)
    answer: str = ", ".join([item["name"] for item in shortest_path[0].nodes])

    task_response = aidevs_send_answer(os.getenv("S03E05_TASK_NAME"), answer)
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )


if __name__ == "__main__":
    run()
