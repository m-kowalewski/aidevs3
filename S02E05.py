import os
import requests
from loguru import logger

from utils import (
    aidevs_send_answer,
    group_files_by_type,
    openai_create,
    openai_vision_create,
    replace_placeholders_in_text,
    transfer_webpage_to_markdown,
    whisper_transcribe,
)


def run():
    url_article: str = os.getenv("S02E05_URL_ARTICLE")
    url_questions: str = os.getenv("S02E05_URL_QUESTIONS")
    output_directory: str = "S02E05"
    markdown_name: str = "S02E05_webpage"
    system_template_vision = """
        You are tasked with analyzing the provided image and describing it in detail. Focus on the following aspects:

            General Overview: Provide a brief summary of what the image depicts.
            Objects and Elements: Identify the key objects, people, or elements visible in the image.
            Context and Setting: Describe the environment, location, or context (e.g., indoor, outdoor, urban, natural).
            Colors and Style: Mention prominent colors, patterns, or artistic style, if relevant.
            Actions or Emotions: Note any actions, interactions, or emotions that are evident in the image.

        Guidelines:

            Be specific and detailed in your description.
            Avoid making assumptions about elements not clearly visible in the image.
            If the image has text, include it in the description.

        Output format:
        Write the description as a complete and coherent paragraph.
        Example Input:

            An image of a park with people sitting on benches and children playing.

        Example Output:

        The image depicts a lively park scene during a sunny day. People are sitting on wooden benches under the shade of tall, green trees, while children are playing on a brightly colored playground in the background. A paved path runs through the park, bordered by well-maintained flower beds with vibrant red and yellow flowers. The sky is clear and blue, adding to the cheerful atmosphere of the setting.
        """
    system_template_chat = """
        ### Prompt:
        You are tasked with answering a user’s question based on the provided context, which includes an article, image descriptions enclosed in `<img></img>` tags, and audio transcriptions enclosed in `<mp3></mp3>` tags.

        ### Guidelines:
        1. Carefully analyze the entire context, including the article, image descriptions, and audio transcriptions.
        2. Focus only on the information relevant to the user’s question.
        3. Respond in **Polish** with a concise, accurate sentence.

        ### Input Example:
        **Context:**
        ```
        Artykuł: Wczoraj w Warszawie odbył się koncert zespołu "Echo". Tłumy zebrały się na placu Defilad, aby wysłuchać ich największych hitów.
        <img>Zdjęcie przedstawia plac pełen ludzi z widoczną sceną i kolorowymi światłami.</img>
        <mp3>Komentator: Koncert rozpoczął się o godzinie 20:00, a wśród hitów nie zabrakło popularnego utworu "Noc w mieście".</mp3>
        ```

        **Question:**
        „O której rozpoczął się koncert?”

        **Output:**
        „Koncert rozpoczął się o godzinie 20:00.”

        ### Instructions:
        - Always answer in **Polish**.
        - Provide only the relevant short sentence as the answer.
        - Do not include unnecessary details or rephrase the question.
        """

    transfer_webpage_to_markdown(url_article, output_directory, markdown_name)

    grouped_files = group_files_by_type(output_directory)
    image_descriptions = dict()
    for image_name in grouped_files["Images"]:
        image = open(os.path.join(output_directory, image_name), "rb")
        response = openai_vision_create(
            system_template_vision, "", [image], model="gpt-4o-mini", temperature=0.1
        )
        logger.debug(f"IMAGE DESCRIPTION: {image_name}\n{response.content}")
        image_descriptions[image_name] = response.content
    audio_transcriptions = dict()
    for audio_name in grouped_files["Audio"]:
        transcription = whisper_transcribe(os.path.join(output_directory, audio_name))
        logger.debug(f"AUDIO TRANSCRIPTION: {audio_name}\n{transcription}")
        audio_transcriptions[audio_name] = transcription

    markdown_path = os.path.join(output_directory, markdown_name)
    with open(markdown_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()
    logger.debug("Successfully opened markdown content")

    webpage_complete_data = replace_placeholders_in_text(
        markdown_content, image_descriptions, audio_transcriptions
    )
    logger.debug("Webpage_complete_data is completed")

    questions_response = requests.get(url_questions)
    questions = questions_response.text
    logger.debug(f"QUESTIONS: {questions}")

    questions_dict = {}
    lines = questions.strip().split("\n")
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)  # Split only at the first '='
            questions_dict[key.strip()] = value.strip()

    answers = {}
    for question_id, question_content in questions_dict.items():
        response = openai_create(
            f"{webpage_complete_data}\n{system_template_chat}", question_content
        )
        answer = response.content
        logger.debug(f"QUESTION: {question_content}\nANSWER: {answer}")
        answers[question_id] = answer
    logger.debug(f"FINAL ANSWERS: {answers}")

    response_task = aidevs_send_answer(
        task=os.getenv("S02E05_TASK_NAME"), answer=answers
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
