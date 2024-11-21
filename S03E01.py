import os
from loguru import logger

from utils import (
    aidevs_send_answer,
    extract_answer,
    group_files_by_type,
    openai_create,
)


def run():
    directory_reports: str = "data/pliki_z_fabryki"
    directory_facts: str = "data/pliki_z_fabryki/facts"
    reports = group_files_by_type(directory_reports, file_types={".txt": "Text"})
    facts = group_files_by_type(directory_facts, file_types={".txt": "Text"})
    context = ""
    reports_content = dict()
    for report in reports["Text"]:
        with open(
            os.path.join(directory_reports, report), "r", encoding="utf-8"
        ) as file:
            content = file.read().strip()
            reports_content[report] = content
            context = f"{context}\n###\n{content}"
    for fact in facts["Text"]:
        with open(os.path.join(directory_facts, fact), "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content == "entry deleted":
                context = f"{context}\n###\n{content}"

    prompt: str = """
        ### Prompt:
        You are tasked with generating keywords based on the user-provided input, using the provided context as a reference. Before providing the final keywords, reflect on what is relevant from the context to the user input.
        
        ### Guidelines:  
        1. The **context** will be enclosed in `<CONTEXT></CONTEXT>` tags and serves only as additional information to help you create accurate and relevant keywords.  
        2. The **user input**, enclosed in `<FILENAME></FILENAME><INPUT></INPUT>` tags, specifies the main focus for the keywords.  
        3. If the context mentions a profession or job title (even if is retired), include it as a keyword (e.g., "nauczyciel", "inżynier").
        4. If the context mentions about technology, include it as a keyword (e.g., "JavaScript", "Python").
        5. If the input mentions about animals, include it as a keyword. (e.g., "zwierzę", "zwierzyna leśna").
        6. Generate keywords in the **Polish language**, ensuring they are in the **nominative case** (e.g., "sportowiec", not "sportowcem" or "sportowców").  
        7. Extract the most relevant and meaningful terms related to the user input while leveraging the context for guidance.  
        8. Separate the keywords with commas.  
        9. Always add sector mentioned in filename. (e.g., "sektor C4", "sektor A3", "sektor A1").
        10. Enclose the final list of keywords in `<ANSWER></ANSWER>` tags.  
        
        ### Example Input:
        ```
        <CONTEXT>Aleksander Ragowski pracował jako nauczyciel języka angielskiego, przez wiele lat prowadząc zajęcia w Szkole Podstawowej nr 9 w Grudziądzu. Ma 34 lata. Jego hobby to bieganie.</CONTEXT>  
        <INPUT>Aleksander Ragowski został złapany przez policję.</INPUT>
        ```
        
        ### Example Output:
        ```
        Znane z kontekstu, mające powiązanie z treścią użytkownika: Aleksader Ragowski był nauczycielem. Pracował w szkole podstawowej. Powiązany jest z Grudziądzem.
        <ANSWER>Aleksander Ragowski, nauczyciel, Grudziądz, szkoła podstawowa, policja, schwytanie</ANSWER>
        ```
    """
    system_template: str = f"{prompt}<CONTEXT>{context}</CONTEXT>"
    logger.debug(f"SYSTEM TEMPLATE: {system_template}")
    answer = dict()
    for report_name, report_content in reports_content.items():
        human_template: str = (
            f"<FILENAME>{report_name}</FILENAME><INPUT>{report_content}</INPUT>"
        )
        response_keywords = openai_create(
            system_template, human_template, model="gpt-4o"
        )
        logger.debug(
            f"Raport name: {report_name}, Response: {response_keywords.content}"
        )
        answer[report_name] = extract_answer(response_keywords.content)
        logger.debug(f"KEYWORDS: {answer[report_name]}")
    logger.info(f"ANSWER DICT: {answer}")

    response_task = aidevs_send_answer(
        task=os.getenv("S03E01_TASK_NAME"), answer=answer
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
