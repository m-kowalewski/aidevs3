import os
import re
from loguru import logger
from typing import List, Optional

from utils import aidevs_send_answer, openai_create, whisper_transcribe


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def run():
    system_template: str = """
    Prompt:
    You are tasked with deducing the street name of the institute where Andrzej Maj teaches, based on six witness testimonies provided by the user.
    
    Guidelines:
    
        Witness statements may contain contradictions, errors, or unusual phrasing, so interpret them carefully.
        The street name is not directly mentioned in any of the transcripts.
        Use your reasoning skills and background knowledge to infer which institute this might be and determine the street name.
        Explain your thought process to support your conclusion.
        Provide your final answer in Polish, enclosed in <answer></answer> tags.
    
    Example output format:
    
    Na podstawie zeznań wydedukowałem, że...  
    <answer>Nazwa Ulicy</answer>
    """
    human_template: str = ""

    directory: str = "data/przesluchania/"
    files: List[str] = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    for file in files:
        transcription = whisper_transcribe(f"{directory}{file}")
        name: str = file.split(".")[0]
        human_template += f"{name}\n{transcription}\n###\n"
    logger.debug(f"HUMAN TEMPLATE: {human_template}")

    response = openai_create(system_template, human_template, model="gpt-4o")
    logger.debug(f"{response.content}")

    answer: str = extract_answer(response.content)
    logger.info(f"ANSWER: {answer}")

    response = aidevs_send_answer(os.getenv("S02E01_TASK_NAME"), answer)
    if response.status_code == 200:
        logger.success(f"Request successful! Response:, {response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {response.status_code}"
            f"Response content: {response.content}"
        )


if __name__ == "__main__":
    run()
