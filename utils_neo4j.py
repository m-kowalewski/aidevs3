import os

from neo4j import GraphDatabase
from typing import List


def create_driver() -> GraphDatabase.driver:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(username, password))


def close_connection(driver: GraphDatabase.driver) -> None:
    driver.close()


# TODO: Functions are not generic.
def add_person(tx, person_id, name) -> None:
    query = """
    MERGE (p:Person {id: $id})
    SET p.name = $name
    """
    tx.run(query, id=person_id, name=name)


def add_relationship(tx, id1, id2) -> None:
    query = """
    MATCH (p1:Person {id: $id1}), (p2:Person {id: $id2})
    MERGE (p1)-[:CONNECTED_TO]->(p2)
    """
    tx.run(query, id1=id1, id2=id2)


def find_shortest_path(session, start_name: str, end_name: str) -> List:
    query = """
        MATCH (start:Person {name: $start_name}), (end:Person {name: $end_name}),
              path = shortestPath((start)-[*]-(end))
        RETURN path
        """
    result = session.run(query, start_name=start_name, end_name=end_name)
    paths = []
    for record in result:
        paths.append(record["path"])
    return paths
