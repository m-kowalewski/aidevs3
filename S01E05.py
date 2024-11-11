import os
import json
import requests
from dotenv import load_dotenv
from loguru import logger
from utils import aidevs_send_answer, generate_local_llm_response


load_dotenv()


def run():
    response = requests.get(os.getenv("S01E05_URL"))
    human_template = response.text
    logger.debug(f"Human template: {human_template}")

    system_template: str = """
    Task: Replace specific personal information in a given text with the word "CENZURA."
    Rules:
    Replace full names with "CENZURA."
    Replace city names with "CENZURA."
    Replace street names and building numbers with "CENZURA."
    Replace ages with "CENZURA."
    Do not alter any other parts of the text.
    Output format: {\"result\":\"..."}"}
    Examples:
    User: 'Dane podejrzanego: Jakub Woźniak. Adres: Rzeszów, ul. Miła 4. Wiek: 33 lata.'
    AI: {\"result\":\"Dane podejrzanego: CENZURA. Adres: CENZURA, ul. CENZURA. Wiek: CENZURA lata."}"}
    
    User: 'Tożsamość podejrzanego: Michał Wiśniewski. Mieszka we Wrocławiu na ul. Słonecznej 20. Wiek: 30 lat.'
    AI: {\"result\":\"Tożsamość podejrzanego: CENZURA. Mieszka we CENZURA na ul. CENZURA. Wiek: CENZURA lat."}"}
    
    User: 'Dane osoby podejrzanej: Paweł Zieliński. Zamieszkały w Warszawie na ulicy Pięknej 5. Ma 28 lat.'
    AI: {\"result\":\"Dane osoby podejrzanej: CENZURA. Zamieszkały w CENZURA na ulicy CENZURA. Ma CENZURA lat."}"}
    
    User: 'Tożsamość osoby podejrzanej: Piotr Lewandowski. Zamieszkały w Łodzi przy ul. Wspólnej 22. Ma 34 lata.'
    AI: {\"result\":\"Tożsamość osoby podejrzanej: CENZURA. Zamieszkały w CENZURA przy ul. CENZURA. Ma CENZURA lata."}"}
    
    User: 'Informacje o podejrzanym: Marek Jankowski. Mieszka w Białymstoku na ulicy Lipowej 9. Wiek: 26 lat.'
    AI: {\"result\":\"Informacje o podejrzanym: CENZURA. Mieszka w CENZURA na ulicy CENZURA. Wiek: CENZURA lat."}"}
    
    User: 'Podejrzany: Krzysztof Kwiatkowski. Mieszka w Szczecinie przy ul. Różanej 12. Ma 31 lat.'
    AI: {\"result\":\"Podejrzany: CENZURA. Mieszka w CENZURA przy ul. CENZURA. Ma CENZURA lat."}"}
    """

    response_llm = generate_local_llm_response(
        system_template=system_template, human_template=human_template
    )
    logger.debug(f"Response_llm: {response_llm}")
    response_llm_dict: dict = json.loads(response_llm)
    data: str = response_llm_dict["result"]
    logger.debug(f"Data: {data}")

    response_task = aidevs_send_answer(task=os.getenv("S01E05_TASK_NAME"), answer=data)
    if response_task.status_code == 200:
        logger.success(f"Request successful! Response:, {response_task.content}")
    else:
        logger.warning(
            f"Request failed with status code: {response_task.status_code}"
            f"Response content: {response_task.content}"
        )


if __name__ == "__main__":
    run()
