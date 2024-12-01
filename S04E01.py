import os
import json
import re
from loguru import logger

from utils_ai import aidevs_send_answer, openai_vision_create
from utils_files_and_text import download_file_from_url, extract_answer


directory: str = "S04E01"


def extract_image_names(text: str) -> list[str]:
    """Extract image names from text using regex pattern.

    Args:
        text: String containing potential image names

    Returns:
        List of found image names matching the pattern IMG_XXX.PNG
    """
    image_pattern = r"IMG_.*?\.PNG"
    return re.findall(image_pattern, text)


def _analyse_image(url_base, image_name):
    system_template_image_analyse: str = """
    ### Updated Prompt:

    You are an advanced vision model designed to analyze images and determine their readability for the user. Based on your assessment, decide if the image is immediately usable or requires enhancement.
    
    ### Guidelines:
    
    1. **Primary Task**:  
       - Examine the input image and assess whether it is readable for the user.  
       - If the image is **readable**, respond with **`<ANSWER>OK</ANSWER>`**.
    
    2. **Enhancement Options**:  
       - If the image is **not readable**, determine the best enhancement action:  
         - **REPAIR**: Use this if the image contains visual glitches, distortions, or noise.  
         - **DARKEN**: Use this if the image is too bright, causing difficulty in viewing details.  
         - **BRIGHTEN**: Use this if the image is too dark, making details unclear.
    
    3. **Decision Process**:  
       - Always provide your reasoning process before your answer. Describe what you observed in the image and why you chose the action.  
       - Ensure your decision aligns with the specific problem detected in the image.
    
    4. **Answer Format**:  
       - Your final decision should be enclosed in `<ANSWER></ANSWER>` tags.  
       - **If you can't make a decision**, return `<ANSWER>OK</ANSWER>` as the default.  
       - Examples:  
         - `<ANSWER>OK</ANSWER>`  
         - `<ANSWER>REPAIR</ANSWER>`  
         - `<ANSWER>DARKEN</ANSWER>`  
         - `<ANSWER>BRIGHTEN</ANSWER>`
    
    ### Example Workflow:
    
    **Input Image 1**:  
    Reasoning: "The image appears clear and readable. There are no visible distortions or lighting issues."  
    **Output**: `<ANSWER>OK</ANSWER>`  
    
    **Input Image 2**:  
    Reasoning: "The image contains visible glitch patterns and noise, making it hard to read. Repairing the image would resolve this issue."  
    **Output**: `<ANSWER>REPAIR</ANSWER>`  
    
    **Input Image 3**:  
    Reasoning: "The image is too dark, making the text and details difficult to discern. Brightening the image will improve readability."  
    **Output**: `<ANSWER>BRIGHTEN</ANSWER>`  
    
    **Input Image 4**:  
    Reasoning: "The image is overly bright, causing glare and washing out important details. Darkening the image will enhance clarity."  
    **Output**: `<ANSWER>DARKEN</ANSWER>`  
    
    ### Notes:  
    
    - Always thoroughly analyze the image and provide a logical explanation for your choice.  
    - Respond with the appropriate action only when needed, and ensure the selected action addresses the image's specific problem.  
    - **If the model cannot make a decision, return `<ANSWER>OK</ANSWER>`**.  
    - Your answer must always be in `<ANSWER></ANSWER>` tags.
    """

    image_url: str = f"{url_base}{image_name}"
    logger.debug(f"Analyzing image: {image_url}")
    download_file_from_url(image_url, directory)
    human_template_analyse: str = ""
    images = [open(f"{directory}/{image_name}", "rb")]
    response_analyse = openai_vision_create(
        system_template_image_analyse, human_template_analyse, images
    )
    logger.debug(response_analyse.content)
    answer: str = extract_answer(response_analyse.content)
    return answer


def run():
    initial_message: str = "START"
    response = aidevs_send_answer(os.getenv("S04E01_TASK_NAME"), initial_message)
    response_message = json.loads(response.text)["message"]
    logger.debug(response_message)
    # TODO: Add logic that finds the URL_BASE in the response message
    url_base = os.getenv("S04E01_URL_BASE")

    image_names = extract_image_names(response_message)
    logger.debug(f"Found images: {image_names}")

    result_images = []

    image_names = image_names[2:]
    for image_name in image_names:
        answer: str = _analyse_image(url_base, image_name)

        count: int = 0
        while answer != "OK":
            count += 1
            if count > 6:
                break
            logger.debug(f"ITERATION: {count}")

            operation: str = f"{answer} {image_name}"
            logger.info(f"OPERATION: {operation}")

            response_order = aidevs_send_answer(
                os.getenv("S04E01_TASK_NAME"), operation
            )
            response_order_message = json.loads(response_order.text)["message"]
            image_name = extract_image_names(response_order_message)[0]
            logger.debug(f"IMAGE NAME: {image_name}")

            answer: str = _analyse_image(url_base, image_name)
        result_images.append(image_name)

    logger.info(f"RESULT IMAGES: {result_images}")

    system_template_description: str = """
    ### Prompt:  

    You are an advanced image analysis model tasked with creating a detailed physical description of a person based on four provided photos. The user supplies 4 images, but not all of them may contain the same person. Your goal is to describe the appearance of the individual from the images that do contain them, in Polish, including all features that may help identify the person.  
    
    ### Guidelines:  
    
    1. **Image Analysis**:  
       - Carefully examine all visible physical features in the photos, such as:  
         - **Wiek** (przybliżony).  
         - **Wzrost** (jeśli możliwy do oszacowania).  
         - **Budowa ciała** (np. szczupła, umięśniona).  
         - **Rysy twarzy** (np. kształt twarzy, nosa, ust, brwi).  
         - **Włosy** (dokładny kolor, długość, fryzura).  
         - **Oczy** (kolor, kształt, szczegóły).  
         - **Znak szczególny** (np. pieprzyk, blizna, tatuaż).  
         - **Ubiór** widoczny na zdjęciach.  
         - **Inne szczegóły**, takie jak okulary, biżuteria czy dodatki.  
    
    2. **Reasoning Process**:  
       - Before providing the final description, share your reasoning process.  
       - For example, you can describe what you noticed in each photo or how you combined details to create the most accurate profile.  
       - If not all images contain the same person, clarify this in your reasoning process.  
    
    3. **Response Format**:  
       - The reasoning process can be written in Polish or English, based on observations.  
       - The final description **must be in Polish** and enclosed in `<ANSWER></ANSWER>` tags.  
       - Begin the description with: *"Na podstawie dostarczonych zdjęć przygotowano następujący rysopis:"*  
       - End the description with: *"Rysopis został sporządzony na podstawie 4 przesłanych zdjęć."*  
    
    ### Example Response:
    
    **Reasoning Process**:  
    After reviewing the four images, I noticed that three of the photos contained the same person, while one photo was unclear. The visible features across the images include consistent hair color and facial structure. One photo also shows a distinct tattoo on the person’s left wrist.  
    
    **Final Description in Tags**:  
    ```  
    <ANSWER>  
    Na podstawie dostarczonych zdjęć przygotowano następujący rysopis:  
    
    - Osoba w wieku około 30-35 lat.  
    - Wzrost szacowany na około 175 cm.  
    - Budowa ciała szczupła.  
    - Twarz owalna, z wyraźnie zaznaczonymi kośćmi policzkowymi. Nos prosty, usta średniej wielkości, brwi gęste i ciemne.  
    - Włosy krótkie, czarne, zaczesane do tyłu.  
    - Oczy brązowe, wąskie, z delikatnymi zmarszczkami wokół.  
    - Widoczny tatuaż na lewym nadgarstku w kształcie smoka.  
    - Ubranie: na zdjęciach osoba nosi ciemnoniebieską kurtkę, białą koszulę i czarne spodnie.  
    - Widoczny zegarek na lewej ręce.  
    
    Rysopis został sporządzony na podstawie 4 przesłanych zdjęć.  
    </ANSWER>  
    ```  
    
    ### Notes:  
    
    - If certain details are unclear, include this in the reasoning process.  
    - Make the description as precise and detailed as possible to help in identification.  
    - Instead of using general terms like "ciemne" for hair color, please use a more precise color, such as czarne, brązowe, blond.
    - If not all images contain the same person, clarify which images were used for the description.  
    - Ensure the description is always enclosed within `<ANSWER></ANSWER>` tags.
    """
    human_template_description: str = ""
    images = [open(f"{directory}/{image_name}", "rb") for image_name in result_images]
    description = openai_vision_create(
        system_template_description,
        human_template_description,
        images,
        model="gpt-4o",
        temperature=0.1,
    )
    logger.debug(description.content)
    answer_description = extract_answer(description.content)
    logger.debug(answer_description)

    task_response = aidevs_send_answer(
        os.getenv("S04E01_TASK_NAME"), answer_description
    )
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )


if __name__ == "__main__":
    run()
