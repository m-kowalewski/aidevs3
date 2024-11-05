import os
import re
import requests
from dotenv import load_dotenv
from loguru import logger
from utils import openai_create


load_dotenv()


def run():
    url = os.getenv("S01E01_URL")
    response = requests.get(url)
    html_content = response.text
    question_match = re.search(
        r'<p id="human-question">Question:<br />(.*?)</p>', html_content
    )
    if question_match:
        question = question_match.group(1).strip()
        logger.success(f"Extracted Question: {question}")
    else:
        logger.warning("Question not found.")
    username = os.getenv("S01E01_USERNAME")
    password = os.getenv("S01E01_PASSWORD")
    system_template = """
    Podaj wyłącznie rok, w którym miało miejsce dane wydarzenie.
    """
    human_template = question
    answer = openai_create(system_template, human_template, model="gpt-4o-mini")
    payload = {
        "username": username,
        "password": password,
        "answer": int(answer.content),
    }
    logger.debug(f"Payload: {payload}")
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        logger.success(f"Request successful! Response:, {response.content}")
    else:
        logger.warning(f"Request failed with status code: {response.status_code}")


if __name__ == "__main__":
    run()
