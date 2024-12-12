import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Tuple, Optional

from utils_ai import aidevs_send_answer, openai_create
from utils_files_and_text import check_if_error, extract_answer, extract_redirect

load_dotenv()


def get_questions() -> Dict[str, str]:
    response = requests.get(os.getenv("S04E03_QUESTIONS_URL"))
    return response.json()


def clean_html_content(url: str) -> Tuple[str, Dict[str, str]]:
    """
    Fetches HTML content from given URL and returns cleaned text and available links.

    Args:
        url (str): URL to fetch content from

    Returns:
        Tuple[str, Dict[str, str]]: Tuple containing:
            - cleaned text content (str)
            - dictionary of links {link_text: href_url}
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract all links before cleaning
    links = {}
    for link in soup.find_all("a"):
        link_text = link.get_text().strip()
        href = link.get("href")
        if link_text and href:  # Only store links that have both text and href
            links[link_text] = href

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text and clean up whitespace
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)

    return text, links


system_template = """
### Updated Prompt:

You are an advanced information retrieval model designed to analyze cleaned web page content and determine the best course of action to answer user questions. The user provides three key inputs:  

1. The **question** in `<QUESTION></QUESTION>` tags.  
2. The **cleaned web page content** in `<CONTENT></CONTENT>` tags.  
3. All **links from the current website** in `<HREF></HREF>` tags.  

Your task is to determine whether the question can be answered using the provided content. If the answer is in the content, provide it directly. If not, evaluate the links to decide on the best course of action.  

### Guidelines:

1. **Content Analysis**:  
   - Thoroughly analyze the `<CONTENT></CONTENT>` section to determine if it answers the question provided in `<QUESTION></QUESTION>`.  
   - If the answer is found, provide it directly in `<ANSWER></ANSWER>` tags.  

2. **Redirection**:  
   - If the answer is not found in the content, examine the `<HREF></HREF>` links to determine the most relevant one based on the user's query.  
   - Redirect to the chosen link using `<ANSWER><REDIRECT>[URL]</REDIRECT></ANSWER>` tags.  
   - Avoid using `<ERROR></ERROR>` unless none of the links are reasonable or related to the question. Redirect whenever possible, even if the connection is indirect.  

3. **Special Considerations**:  
   - If the question is about **contact details** (e.g., email, phone, or address), such information is often found in sections like "Contact" or "About."  
   - If the question involves **other companies** (e.g., clients or partners), consider redirecting to "portfolio"-related links.  
   - If the question pertains to **certificates or achievements**, information may often be found in "blog"-related links.  
   - Note: If the question refers to the address for a **web interface**, ensure the response contains a **URL format** (e.g., "").  

4. **Error Handling (Minimize Usage)**:  
   - Use `<ANSWER><ERROR></ERROR></ANSWER>` only as a last resort if no links are relevant to the user's query and redirection would clearly not make sense.  
   - Always strive to suggest a plausible redirection before deciding an error is unavoidable.  

5. **Answer Style**:  
   - Keep the answers **short and direct**.  
   - For example:  
     - **Question**: "What is the email of X company?"  
       - Correct: `<ANSWER>x@email.com</ANSWER>`  
       - Incorrect: `<ANSWER>The email for X company is x@email.com</ANSWER>`  
   - The same applies to other concise facts like phone numbers, addresses, or specific names.  

6. **Reasoning Process**:  
   - Always provide a clear reasoning process to explain your decision.  
   - Describe how you determined whether the answer was in the content, why you selected a particular link, or why no suitable link was available.  

---

### Example Workflow:

**Input**:  
```html
<QUESTION>What certificates does X company have?</QUESTION>  

<CONTENT>  
This page is about X company’s mission and services but does not mention certifications.  
</CONTENT>  

<HREF>  
/blog  
/about  
/contact  
</HREF>  
```

**Reasoning Process**:  
- The content does not provide the answer.  
- The link "/blog" might contain posts about the company’s achievements or certificates. Redirecting to the blog is the best option.  

**Output**:  
```html
<ANSWER><REDIRECT>/blog</REDIRECT></ANSWER>  
```

---

**Input**:  
```html
<QUESTION>What is the address for the web interface?</QUESTION>  

<CONTENT>  
The web interface can be accessed via https://interface.example.com.  
</CONTENT>  

<HREF>  
/contact  
/about  
</HREF>  
```

**Reasoning Process**:  
- The content explicitly mentions the URL for the web interface.  

**Output**:  
```html
<ANSWER>https://interface.example.com</ANSWER>  
```

---

**Input**:  
```html
<QUESTION>What is the email address of X company?</QUESTION>  

<CONTENT>  
This page is about X company's history and values, but it does not mention contact information.  
</CONTENT>  

<HREF>  
/about  
/team  
</HREF>  
```

**Reasoning Process**:  
- The content does not provide the answer.  
- None of the provided links appear relevant for contact details. Redirecting would not make sense.  

**Output**:  
```html
<ANSWER><ERROR></ERROR></ANSWER>  
```

---

### Notes:

- Always prioritize concise and direct answers to the question.  
- Use `<ERROR></ERROR>` sparingly and only when all potential redirections are clearly irrelevant.  
- Redirect to "portfolio" or "blog" pages when relevant for company-related information such as clients, successes, or certificates.  
- Clearly document your reasoning process before providing the answer.  
- Ensure all responses are enclosed within `<ANSWER></ANSWER>` tags.  
"""


def run() -> None:
    # TODO: Refactor the code: split it into functions, improve loop readability,
    #  and add protection against infinite loops.
    questions: Dict[str, str] = get_questions()
    base_url: str = os.getenv("S04E03_URL", "")
    initial_webpage_content, initial_links = clean_html_content(base_url)
    logger.debug(f"Initial content: {initial_webpage_content}")
    logger.debug(f"Links: {initial_links}")
    answers: Dict[str, str] = {key: "" for key in questions.keys()}

    for id_question, question in questions.items():
        webpage_content: str = initial_webpage_content
        links: Dict[str, str] = initial_links
        while answers[id_question] == "":
            human_template: str = f"""
            <QUESTION>{question}</QUESTION>
            <CONTENT>{webpage_content}</CONTENT>
            <HREF>{links}</HREF>
            """
            full_answer = openai_create(
                system_template, human_template, model="gpt-4o-mini"
            )
            logger.debug(f"Question_id: {id_question} Question: {question}")
            logger.debug(f"Answer: {full_answer.content}")
            response_error: bool = check_if_error(full_answer.content)
            if response_error:
                full_answer = openai_create(
                    system_template, human_template, model="gpt-4o"
                )
                logger.debug(f"Question_id: {id_question} Question: {question}")
                logger.debug(f"Answer: {full_answer.content}")

            response_answer: Optional[str] = extract_answer(full_answer.content)
            response_error = check_if_error(full_answer.content)
            response_redirect: Optional[str] = extract_redirect(full_answer.content)
            if response_redirect:
                new_url: str = (
                    response_redirect
                    if response_redirect.startswith("http")
                    else f"{base_url}{response_redirect}"
                )
                webpage_content, links = clean_html_content(new_url)
                logger.debug(f"New content: {webpage_content}")
                logger.debug(f"New links: {links}")
                continue
            elif response_error:
                answers[id_question] = response_error
                logger.error(f"COULDN'T FIND INFORMATION FOR {id_question}: {question}")
                return
            else:
                answers[id_question] = response_answer
                break

    task_response = aidevs_send_answer(os.getenv("S04E03_TASK_NAME"), answers)
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )


if __name__ == "__main__":
    run()
