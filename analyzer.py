# analyzer.py
import google.generativeai as genai
import os
import json
import time
import requests
import tempfile
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION (loaded from .env) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# --- PROMPT (copied from your script) ---
ANALYSIS_PROMPT = """
**ROLE:** You are an expert video & transcript analyst for Duroflex (furniture & mattress company).
Your task is to analyze both the visual cues and spoken content of this sales call video to extract structured insights.
... [YOUR FULL PROMPT HERE] ...
"""
# --- NOTE: I've truncated the prompt for brevity. Copy your full prompt into the string above. ---

# Helper functions (mostly from your original script)
# [I will place the required helper functions from your script here, slightly modified for clarity]
# ... (download_file, get_filename_and_id_from_url, extract_json_from_text, analyze_video_with_retry) ...

# NOTE: The helper functions (download_file, get_filename_and_id_from_url, etc.) are exactly the same
# as in your original script. I am pasting them here for completeness.

def download_file(url: str, save_path: str, max_retries: int = 3, initial_delay: int = 5) -> bool:
    """Downloads a file from a URL with retries and exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries} for {os.path.basename(save_path)}")
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded {os.path.basename(save_path)}")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed for {os.path.basename(save_path)} (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    logger.error(f"All {max_retries} download attempts failed for {os.path.basename(save_path)}.")
    return False

def get_filename_and_id_from_url(url: str) -> Tuple[str, str]:
    """Extracts a filename and a unique ID from a URL, ensuring a .mp4 extension."""
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        if not path or path == '/':
            unique_id = f"url_{abs(hash(url))}"
            return f"{unique_id}.mp4", unique_id
        filename = os.path.basename(path)
        if not filename:
            filename = f"default_file_{abs(hash(url))}.mp4"
            unique_id = os.path.splitext(filename)[0]
        else:
            unique_id = os.path.splitext(filename)[0]
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.flv')):
            filename += '.mp4'
        return filename, unique_id
    except Exception as e:
        logger.error(f"Error parsing URL '{url}': {e}")
        unique_id = f"error_{abs(hash(url))}"
        return f"{unique_id}.mp4", unique_id

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Safely extracts and parses JSON from the model's response, stripping markdown."""
    text = text.strip()
    if text.startswith('```json'): text = text[len('```json'):].lstrip()
    if text.startswith('```'): text = text[len('```'):].lstrip()
    if text.endswith('```'): text = text[:-len('```')].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON. Error: {e}. Raw text: {text[:500]}...")
        return None

def analyze_video_with_retry(file_path: str, max_analysis_retries: int = 3) -> Optional[Dict[str, Any]]:
    uploaded_file = None
    analysis_attempt = 0
    full_analysis_delay = 60
    while analysis_attempt < max_analysis_retries:
        analysis_attempt += 1
        try:
            logger.info(f"Analysis attempt {analysis_attempt}/{max_analysis_retries} for {os.path.basename(file_path)}")
            logger.info("Uploading file...")
            uploaded_file = genai.upload_file(path=file_path)
            # ... [Rest of your analyze_video_with_retry function] ...
            # The rest of this function is identical to your original script.
            # I am pasting it here for completeness.
            file_status_retries = 0
            max_file_status_retries = 10
            file_status_delay = 30
            while uploaded_file.state.name == "PROCESSING":
                try:
                    logger.info(f"File {uploaded_file.display_name} processing... (check {file_status_retries + 1})")
                    time.sleep(file_status_delay)
                    uploaded_file = genai.get_file(uploaded_file.name)
                    file_status_retries = 0
                    file_status_delay = 30
                except Exception as status_error:
                    logger.warning(f"Failed to get file status: {status_error}")
                    file_status_retries += 1
                    if file_status_retries >= max_file_status_retries:
                        raise Exception("Exceeded max retries for file status check.")
                    logger.info(f"Retrying status check in {file_status_delay * 1.5} seconds...")
                    file_status_delay = int(file_status_delay * 1.5)
            if uploaded_file.state.name == "FAILED":
                raise Exception(f"File processing failed. State: {uploaded_file.state.name}")
            if uploaded_file.state.name != "ACTIVE":
                raise Exception(f"Unexpected file state: {uploaded_file.state.name}")
            logger.info(f"File {uploaded_file.display_name} is active.")
            model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Generating analysis...")
            response = model.generate_content([ANALYSIS_PROMPT, uploaded_file], request_options={'timeout': 1800})
            logger.info("Analysis response received.")
            return extract_json_from_text(response.text)
        except Exception as e:
            logger.error(f"Analysis attempt {analysis_attempt} failed: {e}")
            if uploaded_file:
                try: genai.delete_file(uploaded_file.name)
                except Exception as delete_error: logger.warning(f"Could not delete file after error: {delete_error}")
            if analysis_attempt < max_analysis_retries:
                logger.info(f"Retrying in {full_analysis_delay} seconds...")
                time.sleep(full_analysis_delay)
                full_analysis_delay *= 2
        finally:
            if uploaded_file and uploaded_file.name and uploaded_file.state.name not in ["DELETED", "FAILED"]:
                try: genai.delete_file(uploaded_file.name)
                except Exception as delete_error: logger.warning(f"Could not delete file in cleanup: {delete_error}")
    return None

# --- Main function to be called by the API ---
def process_single_video_url(url: str) -> Dict[str, Any]:
    """
    Takes a single video URL, performs the full analysis pipeline, and returns the result.
    """
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API configured for this task.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return {"error": f"Gemini configuration failed: {e}"}

    original_url = str(url).strip()
    processed_url = original_url

    # Handle relative URLs
    if not urlparse(original_url).scheme in ('http', 'https'):
        if not BASE_URL:
            error_msg = f"Relative URL '{original_url}' found but BASE_URL is not set."
            logger.error(error_msg)
            return {"error": error_msg}
        processed_url = urljoin(BASE_URL, original_url)

    filename, call_id = get_filename_and_id_from_url(processed_url)

    analysis_record = {
        "metadata": {
            "call_id": call_id,
            "recording_url": processed_url,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "model_used": "gemini-1.5-flash"
        },
        "analysis_status": "PENDING",
        "extracted_features": None,
        "error": None
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        local_video_path = os.path.join(temp_dir, filename)
        if download_file(processed_url, local_video_path):
            analysis_record["analysis_status"] = "DOWNLOAD_SUCCESS"
            extracted_data = analyze_video_with_retry(local_video_path)
            if extracted_data:
                analysis_record["extracted_features"] = extracted_data
                analysis_record["analysis_status"] = "ANALYSIS_SUCCESS"
            else:
                analysis_record["analysis_status"] = "ANALYSIS_FAILED"
                analysis_record["error"] = "Gemini analysis failed after retries."
        else:
            analysis_record["analysis_status"] = "DOWNLOAD_FAILED"
            analysis_record["error"] = "Failed to download the video file."
    
    return analysis_record
