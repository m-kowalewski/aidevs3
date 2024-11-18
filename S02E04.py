import os
from loguru import logger
from typing import Dict

from utils import (
    aidevs_send_answer,
    extract_answer,
    group_files_by_type,
    openai_vision_create,
    openai_create,
    whisper_transcribe,
)


def run():
    directory = "data/pliki_z_fabryki"
    files = group_files_by_type(directory=directory)
    system_template_vision = """
    Prompt:
    
    You are tasked with reading and extracting the text from the provided image. Please analyze the image and return the extracted text as it appears, without any modifications.
    Guidelines:
    
        Focus on accurately transcribing the text present in the image.
        Maintain the original formatting and spelling, even if the text is handwritten or difficult to read.
        If the text is unclear, return as much as possible and note any sections that were unreadable.
    
    Output format:
    Return the extracted text directly as a string, enclosed in double quotes.
    Example Input:
    
        Image with handwritten text: "Zgłoszenie awarii systemu - Urządzenia nie działają."
    
    Example Output:
    
        "Zgłoszenie awarii systemu - Urządzenia nie działają."
    """
    system_template_chat = """
    Prompt:
    
    You are tasked with classifying the provided text into one of the following three categories:
    
        Ludzie: The text contains information about captured individuals or evidence of their presence (e.g., traces, belongings, or sightings).
        Hardware: The text describes repaired hardware-related issues (e.g., physical devices, mechanical parts, or electronics) but excludes software-related problems.
        Pozostałe: The text does not fit into either of the above categories.
    
    Guidelines:
    
        Carefully analyze the text and determine which category best fits the information provided.
        If there is ambiguity, prioritize accuracy by considering the primary focus of the text.
        Respond only with the category name: people, hardware, or others in <ANSWER></ANSWER>
    
    Example Inputs and Outputs:
    
    Input:
    „Schwytano podejrzanego na miejscu zdarzenia. Znaleziono jego telefon oraz odciski palców.”
    Output:
    <ANSWER>people</ANSWER>
    
    Input:
    „Naprawiono uszkodzenie płyty głównej w laptopie. Problem był związany z przepalonym kondensatorem.”
    Output:
    <ANSWER>hardware</ANSWER>
    
    Input:
    „Przeprowadzono aktualizację oprogramowania systemowego w celu rozwiązania błędu.”
    Output:
    <ANSWER>others</ANSWER>
    
    Input:
    „Naprawiono skrzynię biegów w uszkodzonym pojeździe.”
    Output:
    <ANSWER>hardware</ANSWER>
    
    Input:
    „Podczas patrolu znaleziono porzucone ubrania oraz dokumenty na nazwisko Kowalski.”
    Output:
    <ANSWER>people</ANSWER>
    """
    text_dict = dict()
    for image_name in files["Images"]:
        image = open(os.path.join(directory, image_name), "rb")
        response = openai_vision_create(
            system_template_vision, "", [image], model="gpt-4o-mini", temperature=0.1
        )
        text_dict[image_name] = response.content
    for audio in files["Audio"]:
        transcription = whisper_transcribe(os.path.join(directory, audio))
        text_dict[audio] = transcription
    for txt in files["Text"]:
        with open(os.path.join(directory, txt), "r", encoding="utf-8") as txt_file:
            text_dict[txt] = txt_file.read()

    result_data = {"people": [], "hardware": [], "others": []}
    for file_name, file_content in text_dict.items():
        human_template = file_content
        response = openai_create(system_template_chat, human_template)
        logger.debug(f"File: {file_name} LLM answer: {response.content}")
        answer = extract_answer(response.content)
        result_data[answer].append(file_name)

    answer_data: Dict[str, str] = {
        key: result_data[key] for key in ["people", "hardware"]
    }
    logger.debug(f"ANSWER DATA: {answer_data}")
    response_task = aidevs_send_answer(
        task=os.getenv("S02E04_TASK_NAME"), answer=answer_data
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
