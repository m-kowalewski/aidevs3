import base64
import os
import requests
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import ImagesResponse
from typing import Any, Dict, List, Literal, Union


load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


def generate_local_llm_response(
    system_template: str,
    human_template: str,
    model: str = "llama2:7b",
    stream: bool = False,
    response_format: str = "json",
    api_url: str = "http://localhost:11434/api/generate",
) -> str:
    payload = {
        "model": model,
        "prompt": human_template,
        "stream": stream,
        "format": response_format,
        "system": system_template,
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", result)
    except requests.exceptions.RequestException as e:
        return f"error: {str(e)}"


def openai_create(
    system_template: str,
    human_template: str,
    model: str = "gpt-4o-mini",
    full_response: bool = False,
) -> Union[Dict[str, Any], str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": human_template},
        ],
    )
    return response if full_response else response.choices[0].message


def openai_vision_create(
    system_template: str,
    human_template: str,
    images: List,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    full_response: bool = False,
) -> Union[Dict[str, Any], str]:
    content = [{"type": "text", "text": human_template}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, {base64.b64encode(image.read()).decode("utf-8")}"
                },
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
    )
    return response if full_response else response.choices[0].message


def openai_image_create(
    human_template: str,
    model: str = "dall-e-3",
    n: int = 1,
    size: Literal[
        "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
    ] = "1024x1024",
) -> ImagesResponse:
    response = client.images.generate(
        model=model, prompt=human_template, n=n, size=size
    )
    return response


def openai_get_embedding(text: str, model: str = "text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def whisper_transcribe(
    path: str, model_name: str = "turbo", full_response: bool = False
) -> Union[Dict[str, Any], str]:
    model = whisper.load_model(model_name)
    result = model.transcribe(path)
    return result if full_response else result["text"]


def aidevs_send_answer(task: str, answer: Any) -> requests.Response:
    apikey: str = os.getenv("AIDEVS3_API_KEY")
    url: str = os.getenv("AIDEVS3_API_URL")
    payload: Dict[str, Any] = {"task": task, "apikey": apikey, "answer": answer}
    return requests.post(url, json=payload)


def aidevs_s03e03_query(user_query: Any) -> requests.Response:
    task: str = os.getenv("S03E03_TASK_NAME")
    apikey: str = os.getenv("AIDEVS3_API_KEY")
    url: str = os.getenv("S03E03_API_URL")
    payload: Dict[str, Any] = {"task": task, "apikey": apikey, "query": user_query}
    return requests.post(url, json=payload)
