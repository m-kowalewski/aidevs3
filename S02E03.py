import os
import requests
from loguru import logger

from utils import aidevs_send_answer, openai_image_create


def run():
    response = requests.get(os.getenv("S02E03_URL"))
    human_template: str = response.text
    logger.debug(f"Human template: {human_template}")
    response = openai_image_create(human_template)
    logger.debug(f"{response}")
    answer: str = response.data[0].url
    logger.info(f"ANSWER: {answer}")
    response_task = aidevs_send_answer(
        task=os.getenv("S02E03_TASK_NAME"), answer=answer
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
