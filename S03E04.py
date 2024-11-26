import json
import os
import requests
from loguru import logger
from utils_ai import aidevs_send_answer, aidevs_s03e04_query, openai_create
from utils_files_and_text import extract_answer


_operation_config = {
    "place": "S03E04_API_PLACES_URL",
    "person": "S03E04_API_PEOPLE_URL",
}


def _send_answer(answer: str):
    task_response = aidevs_send_answer(os.getenv("S03E04_TASK_NAME"), answer)
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
        return None
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )
        return json.loads(task_response.text)["message"]


def run():
    system_template: str = """
    ### Prompt:

    You are tasked with determining the city where BARBARA is currently located. The context contains general information and a story, and you can interact with the environment by asking about cities or people.  
    
    ### Guidelines:
    
    1. **Understand the context**: Analyze the provided context and story carefully to extract initial clues about BARBARA's potential location.  
    2. **Interact strategically**:  
       - If you ask about a **person**, you will receive cities associated with that person's **first name** (only one word is allowed).  
       - If you ask about a **city**, you will receive people connected to that city.  
       - If BARBARA is currently in a specific city, asking about that city will return only "BARBARA." This is a critical confirmation of her location.  
       - **Some questions may be forbidden by the system.** If your question is not allowed, explain this situation in your reasoning and immediately propose an alternative question.  
    3. **Information updates**:  
       - After each iteration, the user will provide updated information in the format:  
         **`Information update: <ANSWER> returns: <RESULT>`**  
         - `<ANSWER>` reflects your previous question.  
         - `<RESULT>` provides the response from the system.  
       - Use this new information to refine your investigation. Take care to avoid repetitive or looping questions.  
    4. **Reason carefully**:  
       - Before each action, provide a clear explanation of your reasoning process based on the context and updated information.  
       - Focus on eliminating possibilities systematically until you can confirm BARBARA's location.  
    5. **Provide your answer in the required format**:  
       - Your answer must be inside `<ANSWER></ANSWER>` tags.  
       - Use **`person`** for actions involving asking about a specific first name.  
       - Use **`place`** for actions involving asking about a specific city.  
       - Use **`solution`** when you are confident you know where BARBARA is located.  
    6. **Word format rules**: Answers in `<ANSWER>` must:  
       - Be two words separated by a comma.  
       - Use **CAPITAL letters only**.  
       - Exclude Polish diacritics and be in the **mianownik** form.  
    7. **Avoid loops**: Carefully track previous questions and answers to ensure you do not repeat the same inquiries or get stuck in a cycle.  
    
    ### Key Detail:  
    - If asking about a city returns **only "BARBARA"**, this is definitive proof that BARBARA is in that city. Use this signal to confidently declare the solution.  
    
    ### Example Process:
    
    1. Initial reasoning:  
       - "The context mentions RAFAL and BARBARA. RAFAL might provide insights into BARBARA's location. I will ask about RAFAL."  
       - **Output**: `<ANSWER>person, RAFAL</ANSWER>`  
    
    2. Information update:  
       - **`Information update: person, RAFAL returns: KRAKOW LUBLIN WARSZAWA`**  
       - Reasoning: "RAFAL is connected to three cities. To narrow it down, I will ask about KRAKOW to see who else is connected to it."  
       - **Output**: `<ANSWER>place, KRAKOW</ANSWER>`  
    
    3. Information update:  
       - **`Information update: place, KRAKOW returns: BARBARA ANNA MARCIN`**  
       - Reasoning: "BARBARA is connected to KRAKOW, but this doesn't confirm she's currently there. I will now ask about LUBLIN to explore further."  
       - **Output**: `<ANSWER>place, LUBLIN</ANSWER>`  
    
    4. Information update:  
       - **`Information update: place, LUBLIN returns: MARCIN ALEKSANDER`**  
       - Reasoning: "BARBARA is not mentioned in LUBLIN. This makes KRAKOW more likely. I will confirm by asking about WARSZAWA."  
       - **Output**: `<ANSWER>place, WARSZAWA</ANSWER>`  
    
    5. Information update:  
       - **`Information update: place, WARSZAWA returns: BARBARA`**  
       - Reasoning: "The response for WARSZAWA returns only 'BARBARA,' which confirms she is currently located in WARSZAWA."  
       - **Output**: `<ANSWER>solution, WARSZAWA</ANSWER>`  
    
    ### Notes:  
    - Always provide a detailed reasoning process before submitting any `<ANSWER>` tag.  
    - Use updated information effectively to progress logically and avoid redundant or looping questions.  
    - The final answer must confidently identify BARBARA's location.
    """
    introduction_data_response = requests.get(url=os.getenv("S03E04_BARBARA_URL"))
    introduction_data: str = introduction_data_response.text
    human_template: str = f"{introduction_data}"
    for i in range(35):
        response = openai_create(system_template, human_template, model="gpt-4o")
        # logger.debug(f"Full response: {response.content}")
        operation, query = extract_answer(response.content).split(", ")
        logger.debug(f"OPERATION: {operation}, QUERY: {query}")
        if operation == "solution":
            logger.info(f"FOUND SOLUTION! {response.content}")
            response_query_message = _send_answer(query)
            if not response_query_message:
                break
        else:
            response_query = aidevs_s03e04_query(query, _operation_config[operation])
            response_query_message = json.loads(response_query.text)["message"]
        logger.debug(f"AIDEVS RESPONSE: {response_query_message}")
        human_template = f"{human_template}\nInformation update: {operation}, {query} returns: {response_query_message}"


if __name__ == "__main__":
    run()
