import os
import json
from dotenv import load_dotenv
from loguru import logger
from utils_ai import openai_create, aidevs_send_answer


load_dotenv()


def run():
    system_template = "Answer the question in one word."
    with open("data/S01E03.json", "r") as file:
        data = json.load(file)
    for i, element in enumerate(data["test-data"]):
        num1, operator, num2 = element["question"].split()
        result = int(num1) + int(num2)
        if element["answer"] != result:
            logger.info(
                f"Element: {i}"
                f"Calculation: {element['question']}"
                f"Wrong value: {element['answer']}"
                f"Correct value: {result}"
            )
            data["test-data"][i]["answer"] = result
        if element.get("test"):
            human_template = element["test"]["q"]
            llm_answer = openai_create(system_template, human_template)
            data["test-data"][i]["test"]["a"] = llm_answer.content
            logger.info(
                f"Element: {i}"
                f"Question: {human_template}"
                f"LLM_answer: {llm_answer}"
            )
    data["apikey"] = os.getenv("AIDEVS3_API_KEY")

    with open("output.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    response = aidevs_send_answer(task=os.getenv("S01E03_TASK_NAME"), answer=data)
    if response.status_code == 200:
        logger.success(f"Request successful! Response:, {response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {response.status_code}"
            f"Response content: {response.content}"
        )


if __name__ == "__main__":
    run()
