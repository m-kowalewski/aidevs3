import json
import os
from loguru import logger
from typing import Any, Dict, List

from utils_ai import aidevs_send_answer, openai_create


directory: str = "data/lab_data"
system_template: str = (
    "Classify content of this message as either correct or incorrect."
)


def create_json_input():
    input_files = ["correct.txt", "incorrect.txt"]

    # Process each input file
    for file_name in input_files:
        result_json: List[Dict[str, Any]] = []
        with open(os.path.join(directory, file_name), "r") as file:
            for line in file:
                line = line.strip()
                logger.debug(line)
                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": system_template},
                    {"role": "user", "content": line},
                    {"role": "assistant", "content": file_name.split(".")[0]},
                ]
                result_json.append({"messages": messages})

        logger.debug(result_json)
        output_file = f"{file_name.split('.')[0]}.jsonl"
        with open(os.path.join(directory, output_file), "w") as f:
            for item in result_json:
                f.write(json.dumps(item) + "\n")

    # Combine into finetuning.jsonl
    combined_data = []
    for input_file in ["correct.jsonl", "incorrect.jsonl"]:
        with open(os.path.join(directory, input_file), "r") as f:
            for line in f:
                combined_data.append(line.strip())

    # Write combined data to finetuning.jsonl
    with open(os.path.join(directory, "finetuning.jsonl"), "w") as f:
        for line in combined_data:
            f.write(line + "\n")

    logger.success(f"Created finetuning.jsonl with {len(combined_data)} examples")


def run():
    input_file: str = "verify.txt"
    result_answers: List[str] = []
    with open(os.path.join(directory, input_file), "r") as file:
        for line in file:
            line = line.strip()
            logger.debug(line)
            human_template: str = f"{line.split('=')[1]}"
            assistant_response: str = openai_create(
                system_template,
                human_template,
                model=os.getenv("S04E02_MODEL"),
            )
            logger.debug(assistant_response.content)
            if assistant_response.content == "correct":
                result_answers.append(line.split("=")[0])

    logger.debug(result_answers)

    task_response = aidevs_send_answer(os.getenv("S04E02_TASK_NAME"), result_answers)
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )


if __name__ == "__main__":
    # create_json_input()
    run()
