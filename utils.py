import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict, Union


load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


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
