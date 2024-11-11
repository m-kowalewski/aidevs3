import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict, Union


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


def aidevs_send_answer(task: str, answer: str) -> requests.Response:
    apikey: str = os.getenv("AIDEVS3_API_KEY")
    url: str = os.getenv("AIDEVS3_API_URL")
    payload: Dict[str, Any] = {"task": task, "apikey": apikey, "answer": answer}
    return requests.post(url, json=payload)
