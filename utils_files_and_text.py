import os
import re
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from loguru import logger
from typing import Dict, List, Optional
from urllib.parse import urljoin


def transfer_webpage_to_markdown(url: str, output_dir: str, markdown_name: str) -> None:
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Fetch the HTML content of the page
        response: requests.Response = requests.get(url)
        response.raise_for_status()
        html_content: str = response.text

        # Parse the HTML content with BeautifulSoup
        soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")

        # Initialize Markdown content
        markdown_content: list[str] = []

        # Replace images
        img: Optional[BeautifulSoup.Tag]
        for img in soup.find_all("img"):
            img_url: str = urljoin(url, img.get("src", ""))
            img_name: str = os.path.basename(img_url)
            download_file_from_url(img_url, output_dir)
            img.replace_with(f"<img>{img_name}</img>")

        # Replace MP3 links
        link: Optional[BeautifulSoup.Tag]
        for link in soup.find_all(
            "a", href=lambda href: href and href.endswith(".mp3")
        ):
            mp3_url: str = urljoin(url, link.get("href", ""))
            mp3_name: str = os.path.basename(mp3_url)
            download_file_from_url(mp3_url, output_dir)
            link.replace_with(f"<audio>{mp3_name}</audio>")

        # Extract and append the modified HTML content as Markdown
        markdown_content.append(soup.get_text())

        # Save the Markdown content to a file
        markdown_file: str = os.path.join(output_dir, markdown_name)
        with open(markdown_file, "w", encoding="utf-8") as file:
            file.write("\n".join(markdown_content))

        logger.info(f"Markdown content saved to {markdown_file}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")


def replace_placeholders_in_text(
    text: str,
    image_descriptions: Dict[str, str],
    audio_transcriptions: Dict[str, str],
) -> str:
    def _replace_image_placeholder(match):
        image_name = match.group(1)
        description = image_descriptions.get(image_name, "")
        return f"<img>{description}</img>" if description else match.group(0)

    def _replace_audio_placeholder(match):
        audio_name = match.group(1)
        transcription = audio_transcriptions.get(audio_name, "")
        return f"<mp3>{transcription}</mp3>" if transcription else match.group(0)

    image_pattern = r"<img>(.+?)</img>"
    audio_pattern = r"<audio>(.+?)</audio>"

    text = re.sub(image_pattern, _replace_image_placeholder, text)
    text = re.sub(audio_pattern, _replace_audio_placeholder, text)

    return text


def download_file_from_url(file_url: str, output_dir: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified directory.
    Args:
        file_url (str): The URL of the file to download.
        output_dir (str): Directory to save the downloaded file.
    Returns:
        None
    """
    try:
        # Fetch the file
        response: requests.Response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Extract the file name from the URL
        file_name: str = os.path.basename(file_url)
        file_path: str = os.path.join(output_dir, file_name)

        # Write the file content to the output directory
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded: {file_name}")
    except Exception as e:
        logger.warning(f"Failed to download {file_url}: {e}")


def group_files_by_type(
    directory: str,
    file_types: Dict[str, str] = {".png": "Images", ".mp3": "Audio", ".txt": "Text"},
) -> Dict[str, List[str]]:
    """
    Groups files in the given directory by their type (.png, .mp3, .txt).
    Args:
        directory (str): The path to the directory to scan.
        file_types (Dict[str, str]): A dictionary mapping file extensions (e.g., ".png") to category names
            (e.g., "Images"). Default is {".png": "Images", ".mp3": "Audio", ".txt": "Text"}.
    Returns:
        Dict[str, List[str]]: A dictionary where keys are file types and values are lists of file paths.
    """
    grouped_files = defaultdict(list)

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_name)
            if ext in file_types:
                grouped_files[file_types[ext]].append(file_name)

    return grouped_files


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_redirect(text: str) -> Optional[str]:
    match = re.search(r"<REDIRECT>(.*?)</REDIRECT>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def check_if_error(text: str) -> bool:
    match = re.search(r"<ERROR>(.*?)</ERROR>", text, re.DOTALL)
    return True if match else False
