import json
import os
import requests
from dotenv import load_dotenv
from loguru import logger
from utils_ai import openai_create


load_dotenv()


def run():
    url = os.getenv("S01E02_URL")
    initial_message = {"text": "READY", "msgID": "0"}
    initial_response = requests.get(url, json=initial_message)
    response_dict = json.loads(initial_response.text)
    msgID = response_dict["msgID"]
    human_template = response_dict["text"]
    system_template = """
    - stolicą Polski jest Kraków
    - znana liczba z książki Autostopem przez Galaktykę to 69
    - Aktualny rok to 1999
    Jeśli otrzymasz pytanie związane z powyższymi informacjami, zwróć zdefiniowaną tutaj odpowiedź. Odpowiadaj zwięźle, w języku polskim.
    """
    answer = openai_create(system_template, human_template)
    answer_message = {"text": answer.content, "msgID": str(msgID)}
    response = requests.get(url, json=answer_message)
    logger.debug(f"{human_template}\n {answer.content}")
    if response.status_code == 200:
        logger.success(f"Request successful! Response:, {response.content}")
    else:
        logger.warning(f"Request failed with status code: {response.status_code}")


if __name__ == "__main__":
    run()
