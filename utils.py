import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


def openai_create(
    system_template, human_template, model="gpt-4o-mini", full_response=False
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": human_template},
        ],
    )
    return response if full_response else response.choices[0].message
