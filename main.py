# main.py
# Final version for GCP Cloud Run Deployment

# 1. IMPORTS
import os
import json
import time
import requests
import tempfile
import re
import logging
import uuid
import pandas as pd
import io

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, Tuple

# Import the Google Cloud client library for Secret Manager
from google.cloud import secretmanager

# 2. CONFIGURATION & LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FUNCTION TO FETCH SECRET FROM GCP SECRET MANAGER ---
def get_google_api_key():
    """
    Fetches the Google API Key from GCP Secret Manager.
    This is the secure, production-ready way to handle secrets on GCP.
    """
    try:
        # Create the Secret Manager client
        client = secretmanager.SecretManagerServiceClient()
        
        # Get the project ID from the environment variable provided by Cloud Run
        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            logger.error("GCP_PROJECT environment variable not set. Cannot fetch secret.")
            return None
        
        # Build the resource name of the secret's latest version
        name = f"projects/{project_id}/secrets/GOOGLE_API_KEY/versions/latest"
        
        # Access the secret version
        response = client.access_secret_version(request={"name": name})
        
        # Decode the secret payload
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"FATAL: Could not fetch GOOGLE_API_KEY from Secret Manager: {e}")
        # Return None or raise an exception to prevent the app from running without a key
        return None

# --- LOAD SECRETS AND SET CONSTANTS ---
GOOGLE_API_KEY = get_google_api_key()
# Load non-secret config from environment variables (set in Cloud Run UI)
BASE_URL = os.getenv("BASE_URL", "https://store.popin.to") # Default value is a fallback
URL_COLUMN_NAME = 'Recorded File'

ANALYSIS_PROMPT = """
ROLE: You are an expert video & transcript analyst for Duroflex (furniture & mattress company).
Your task is to analyze both the visual cues and spoken content of this sales call video to extract structured insights.

IMPORTANT INSTRUCTIONS:

Focus ONLY on the requested features. Ignore anything else.

Base your analysis STRICTLY on observable video (visual) and audio (spoken) evidence.

For RELAX, mark each step as:

done: true/false

compliance_score: 1–5 (1 = Not done, 3 = Partially done, 5 = Fully done as per script)

compliance_notes: Short 1–2 line explanation

Output MUST be valid JSON in the exact structure below. No extra commentary.

FEATURES TO EXTRACT:

Visual Analysis (Body Orientation):

customer_orientation: Vertical / Horizontal

agent_orientation_initial: Vertical / Horizontal

agent_orientation_demo: Vertical / Horizontal

Product Demonstration:

agent_showed_product: true/false (true only if a core product is clearly shown: Mattress, Sofa, Bed, Recliner, Pillow, Dining Set).

demo_quality_rating: 1–5

demo_quality_reason: Brief explanation for the rating.

Conversation Analysis:

product_of_interest: Product mentioned by customer (or "Not specified").

call_objective: Primary purpose (e.g., "Product Inquiry", "Price Negotiation", "Purchase Decision", "Post-Purchase Query").

key_themes: Main discussion topics (e.g., ["comfort", "price & EMI", "warranty", "delivery", "store visit invitation"]).

opportunity_for_demo: true/false (whether there was an opportunity for agent to demonstrate a product).

opportunity_for_store_visit: true/false (whether there was an opportunity for agent to suggest store visit).

agent_asked_store_visit: true/false (whether the agent explicitly asked customer for store visit).

RELAX Framework Adherence:

reach_out: {
"done": true/false,
"compliance_score": 1–5,
"compliance_notes": "Short description of how greeting/audio check was handled."
}

explore_needs: {
"done": true/false,
"compliance_score": 1–5,
"compliance_notes": "Short description of how customer need was explored."
}

link_demo: {
"done": true/false,
"compliance_score": 1–5,
"compliance_notes": "Short description of how product demo linked to need."
}

add_value: {
"done": true/false,
"compliance_score": 1–5,
"compliance_notes": "Short description of whether add-ons were suggested."
}

express_value: {
"done": true/false,
"compliance_score": 1–5,
"compliance_notes": "Short description of whether offers, EMI, and store visit were mentioned."
}

Customer Signals:

interest_level: One of ["High", "Medium", "Low"].

objections: List of concerns raised (e.g., ["too expensive", "undecided"]).

next_step: Most likely outcome (e.g., "Will visit store", "Needs follow-up", "Interested but undecided", "No interest").

agent_video_on_start: true/false (whether agent’s video was on from the beginning).

OUTPUT JSON STRUCTURE:

{
"visual_analysis": {
"customer_orientation": "Vertical",
"agent_orientation_initial": "Vertical",
"agent_orientation_demo": "Vertical"
},
"product_demo_analysis": {
"agent_showed_product": true,
"demo_quality_rating": 4,
"demo_quality_reason": "Agent sat on mattress and explained its back support features clearly."
},
"conversation_analysis": {
"product_of_interest": "Duroflex Back Magic Pro Orthopaedic Mattress",
"call_objective": "Product Inquiry",
"key_themes": ["comfort", "warranty", "EMI", "delivery"],
"opportunity_for_demo": true,
"opportunity_for_store_visit": true,
"agent_asked_store_visit": true
},
"relax_framework": {
"reach_out": {
"done": true,
"compliance_score": 5,
"compliance_notes": "Agent greeted warmly and checked audio before starting."
},
"explore_needs": {
"done": true,
"compliance_score": 5,
"compliance_notes": "Agent asked customer’s name and confirmed interest in recliners."
},
"link_demo": {
"done": true,
"compliance_score": 5,
"compliance_notes": "Agent demonstrated recliner features and linked them to back support concerns."
},
"add_value": {
"done": false,
"compliance_score": 1,
"compliance_notes": "No add-ons like pillows or protectors were mentioned."
},
"express_value": {
"done": true,
"compliance_score": 4,
"compliance_notes": "Agent shared EMI options and invited the customer to visit the store."
}
},
"customer_signals": {
"interest_level": "Medium",
"objections": ["too expensive"],
"next_step": "Will visit store",
"agent_video_on_start": true
}
}

# 3. HELPER FUNCTIONS (Analysis Logic)

def flatten_analysis_data(data: dict) -> dict:
    """Flattens the nested JSON structure into a single-level dictionary."""
    flattened = {}
    if data.get("metadata"):
        for k, v in data["metadata"].items():
            flattened[f"metadata_{k}"] = v
    flattened["analysis_status"] = data.get("analysis_status")
    flattened["error"] = data.get("error")
    if data.get("extracted_features"):
        extracted = data["extracted_features"]
        if extracted.get("visual_analysis"):
            for k, v in extracted["visual_analysis"].items():
                flattened[f"visual_{k}"] = v
        if extracted.get("product_demo_analysis"):
            for k, v in extracted["product_demo_analysis"].items():
                flattened[f"demo_{k}"] = v
        if extracted.get("conversation_analysis"):
            for k, v in extracted["conversation_analysis"].items():
                flattened[f"conversation_{k}"] = "; ".join(map(str, v)) if isinstance(v, list) else v
        if extracted.get("relax_framework"):
            for stage, details in extracted["relax_framework"].items():
                if isinstance(details, dict):
                    for k, v in details.items():
                        flattened[f"relax_{stage}_{k}"] = v
        if extracted.get("customer_signals"):
            for k, v in extracted["customer_signals"].items():
                flattened[f"customer_{k}"] = "; ".join(map(str, v)) if isinstance(v, list) else v
    return flattened

def sanitize_filename(filename: str) -> str:
    """Removes characters that are illegal in filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename).strip()

def get_filename_and_id_from_url(url: str) -> Tuple[str, str]:
    """Extracts a sanitized filename and a unique ID from a URL."""
    try:
        path = urlparse(url).path
        base_filename = os.path.basename(path) if path and os.path.basename(path) else f"url_{abs(hash(url))}"
        unique_id = os.path.splitext(base_filename)[0]
        safe_filename = sanitize_filename(base_filename)
        if not safe_filename: safe_filename = "default.mp4"
        if not safe_filename.lower().endswith(('.mp4', '.mov', '.avi')):
            safe_filename += '.mp4'
        return safe_filename, unique_id
    except Exception as e:
        logger.error(f"Error parsing URL '{url}': {e}")
        unique_id = f"error_{abs(hash(url))}"
        return f"{unique_id}.mp4", unique_id

def download_file(url: str, save_path: str) -> bool:
    """Downloads a file from a URL."""
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        logger.info(f"Successfully downloaded {os.path.basename(save_path)}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {os.path.basename(save_path)}: {e}")
        return False

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Safely extracts and parses JSON from the model's response."""
    text = text.strip()
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match: text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON. Raw text: {text[:500]}...")
        return None

def analyze_video(file_path: str) -> Optional[Dict[str, Any]]:
    """Uploads, analyzes, and cleans up a single video file with Gemini."""
    import google.generativeai as genai
    uploaded_file = None
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not configured. Analysis cannot proceed.")
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info(f"Uploading {os.path.basename(file_path)}...")
        uploaded_file = genai.upload_file(path=file_path)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(10)
            uploaded_file = genai.get_file(uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            raise Exception("Google API file processing failed.")
        
        logger.info("Generating analysis from model...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([ANALYSIS_PROMPT, uploaded_file], request_options={'timeout': 1800})
        return extract_json_from_text(response.text)
    except Exception as e:
        logger.error(f"Analysis failed for {os.path.basename(file_path)}: {e}")
        return None
    finally:
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                logger.info(f"Deleted uploaded file {uploaded_file.name}.")
            except Exception as delete_error:
                logger.warning(f"Could not delete file {uploaded_file.name}: {delete_error}")

def process_single_video_url(url: str) -> Dict[str, Any]:
    """Main pipeline for processing one URL from download to analysis."""
    processed_url = url.strip()
    if not urlparse(processed_url).scheme:
        processed_url = urljoin(BASE_URL, processed_url)
    
    filename, call_id = get_filename_and_id_from_url(processed_url)
    analysis_record = {
        "metadata": {"call_id": call_id, "recording_url": processed_url},
        "analysis_status": "PENDING", "extracted_features": None, "error": None
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, filename)
        if download_file(processed_url, local_path):
            extracted_data = analyze_video(local_path)
            if extracted_data:
                analysis_record["extracted_features"] = extracted_data
                analysis_record["analysis_status"] = "ANALYSIS_SUCCESS"
            else:
                analysis_record["analysis_status"] = "ANALYSIS_FAILED"
                analysis_record["error"] = "Gemini analysis failed or returned invalid JSON."
        else:
            analysis_record["analysis_status"] = "DOWNLOAD_FAILED"
            analysis_record["error"] = "Failed to download the video file."
    return analysis_record

# 4. BACKGROUND WORKERS

def run_batch_analysis(batch_id: str, file_contents: bytes, filename: str, tasks_db: Dict[str, Any]):
    """Background worker to process an entire uploaded file."""
    try:
        df = pd.read_csv(io.BytesIO(file_contents)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_contents))
        if URL_COLUMN_NAME not in df.columns:
            raise ValueError(f"File must contain a column named '{URL_COLUMN_NAME}'.")
        
        urls = df[URL_COLUMN_NAME].dropna().tolist()
        total = len(urls)
        tasks_db[batch_id].update({"status": "processing", "total_urls": total, "processed_count": 0, "results": [], "progress": 0.0})

        for i, url in enumerate(urls):
            result = process_single_video_url(url)
            tasks_db[batch_id]["results"].append(flatten_analysis_data(result))
            tasks_db[batch_id]["processed_count"] = i + 1
            tasks_db[batch_id]["progress"] = round(((i + 1) / total) * 100, 2)
        
        tasks_db[batch_id]["status"] = "completed"
    except Exception as e:
        tasks_db[batch_id]["status"] = "failed"
        tasks_db[batch_id]["error"] = str(e)

def run_single_url_analysis(batch_id: str, url: str, tasks_db: Dict[str, Any]):
    """Background worker to process a single URL."""
    try:
        tasks_db[batch_id].update({"status": "processing", "total_urls": 1, "processed_count": 0, "results": [], "progress": 0.0})
        result = process_single_video_url(url)
        tasks_db[batch_id]["results"].append(flatten_analysis_data(result))
        tasks_db[batch_id]["processed_count"] = 1
        tasks_db[batch_id]["progress"] = 100.0
        tasks_db[batch_id]["status"] = "completed"
    except Exception as e:
        tasks_db[batch_id]["status"] = "failed"
        tasks_db[batch_id]["error"] = str(e)

# 5. FASTAPI APPLICATION

app = FastAPI(title="Video Analysis API", version="3.3.0-gcp")
tasks_db: Dict[str, Dict[str, Any]] = {}

class SingleUrlRequest(BaseModel):
    video_url: str

@app.post("/analyze-batch", status_code=202)
async def start_batch_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a file, creates a task, and starts batch analysis."""
    batch_id = str(uuid.uuid4())
    contents = await file.read()
    tasks_db[batch_id] = {"status": "accepted"}
    background_tasks.add_task(run_batch_analysis, batch_id, contents, file.filename, tasks_db)
    return {"batch_id": batch_id, "message": "Batch analysis accepted."}

@app.post("/analyze-url", status_code=202)
async def start_single_url_analysis(request: SingleUrlRequest, background_tasks: BackgroundTasks):
    """Accepts a single URL and starts analysis."""
    batch_id = str(uuid.uuid4())
    tasks_db[batch_id] = {"status": "accepted"}
    background_tasks.add_task(run_single_url_analysis, batch_id, request.video_url, tasks_db)
    return {"batch_id": batch_id, "message": "Single URL analysis accepted."}

@app.get("/batch-status/{batch_id}")
def get_batch_status(batch_id: str):
    """Retrieves the status of any job."""
    task = tasks_db.get(batch_id)
    if not task:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return {k: v for k, v in task.items() if k != 'results'}

@app.get("/results/{batch_id}")
async def get_batch_results(batch_id: str):
    """Returns the results of any completed job as a CSV."""
    task = tasks_db.get(batch_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job not found or not complete. Status: {task.get('status')}")
    if not task.get("results"):
         raise HTTPException(status_code=404, detail="No results found for this job.")
    results_df = pd.DataFrame(task["results"])
    stream = io.StringIO()
    results_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=analysis_results_{batch_id}.csv"
    return response
