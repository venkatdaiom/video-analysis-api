# analyzer.py
import google.generativeai as genai
import os, json, time, requests, tempfile, re, logging
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = os.getenv("BASE_URL")
URL_COLUMN_NAME = 'Recorded File' # The name of the column in your CSV with the URLs

ANALYSIS_PROMPT = """
**ROLE:** You are an expert video & transcript analyst for Duroflex (furniture & mattress company).
Your task is to analyze both the visual cues and spoken content of this sales call video to extract structured insights.
**IMPORTANT INSTRUCTIONS:**
1. Focus ONLY on the requested features. Ignore anything else.
2. Base your analysis STRICTLY on observable video (visual) and audio (spoken) evidence.
3. For RELAX, mark each step as:
   - `done`: true/false
   - `compliance_score`: 1–5 (1 = Not done, 3 = Partially done, 5 = Fully done as per script)
   - `compliance_notes`: Short 1–2 line explanation
4. Output MUST be valid JSON in the exact structure below. No extra commentary.
---
**FEATURES TO EXTRACT:**
1. **Visual Analysis (Body Orientation):**
   * `customer_orientation`: Vertical / Horizontal
   * `agent_orientation_initial`: Vertical / Horizontal
   * `agent_orientation_demo`: Vertical / Horizontal
2. **Product Demonstration:**
   * `agent_showed_product`: true/false (true only if a core product is clearly shown: Mattress, Sofa, Bed, Recliner, Pillow, Dining Set).
   * `demo_quality_rating`: 1–5
   * `demo_quality_reason`: Brief explanation for the rating.
3. **Conversation Analysis:**
   * `product_of_interest`: Product mentioned by customer (or "Not specified").
   * `call_objective`: Primary purpose (e.g., "Product Inquiry", "Price Negotiation", "Purchase Decision", "Post-Purchase Query").
   * `key_themes`: Main discussion topics (e.g., ["comfort", "price & EMI", "warranty", "delivery", "store visit invitation"]).
4. **RELAX Framework Adherence:**
   * `reach_out`: { "done": true/false, "compliance_score": 1–5, "compliance_notes": "Short description of how greeting/audio check was handled." }
   * `explore_needs`: { "done": true/false, "compliance_score": 1–5, "compliance_notes": "Short description of how customer need was explored." }
   * `link_demo`: { "done": true/false, "compliance_score": 1–5, "compliance_notes": "Short description of how product demo linked to need." }
   * `add_value`: { "done": true/false, "compliance_score": 1–5, "compliance_notes": "Short description of whether add-ons were suggested." }
   * `express_value`: { "done": true/false, "compliance_score": 1–5, "compliance_notes": "Short description of whether offers, EMI, and store visit were mentioned." }
5. **Customer Signals:**
   * `interest_level`: One of ["High", "Medium", "Low"].
   * `objections`: List of concerns raised (e.g., ["too expensive", "undecided"]).
   * `next_step`: Most likely outcome (e.g., "Will visit store", "Needs follow-up", "Interested but undecided", "No interest").
---
**OUTPUT JSON STRUCTURE:**
{
  "visual_analysis": { "customer_orientation": "Vertical", "agent_orientation_initial": "Vertical", "agent_orientation_demo": "Vertical" },
  "product_demo_analysis": { "agent_showed_product": true, "demo_quality_rating": 4, "demo_quality_reason": "Agent sat on mattress and explained its back support features clearly." },
  "conversation_analysis": { "product_of_interest": "Duroflex Back Magic Pro Orthopaedic Mattress", "call_objective": "Product Inquiry", "key_themes": ["comfort", "warranty", "EMI", "delivery"] },
  "relax_framework": {
    "reach_out": { "done": true, "compliance_score": 5, "compliance_notes": "Agent greeted warmly and checked audio before starting." },
    "explore_needs": { "done": true, "compliance_score": 5, "compliance_notes": "Agent asked customer’s name and confirmed interest in recliners." },
    "link_demo": { "done": true, "compliance_score": 5, "compliance_notes": "Agent demonstrated recliner features and linked them to back support concerns." },
    "add_value": { "done": false, "compliance_score": 1, "compliance_notes": "No add-ons like pillows or protectors were mentioned." },
    "express_value": { "done": true, "compliance_score": 4, "compliance_notes": "Agent shared EMI options and invited the customer to visit the store." }
  },
  "customer_signals": { "interest_level": "Medium", "objections": ["too expensive"], "next_step": "Will visit store" }
}
"""

def sanitize_filename(filename: str) -> str:
    """Removes characters that are illegal in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', filename).strip()

def get_filename_and_id_from_url(url: str) -> Tuple[str, str]:
    """Extracts a sanitized filename and a unique ID from a URL."""
    try:
        path = urlparse(url).path
        base_filename = os.path.basename(path) if path else f"url_{abs(hash(url))}"
        unique_id = os.path.splitext(base_filename)[0]
        safe_filename = sanitize_filename(base_filename)
        if not safe_filename.lower().endswith(('.mp4', '.mov')):
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
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {os.path.basename(save_path)}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {os.path.basename(save_path)}: {e}")
        return False

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Safely extracts and parses JSON from the model's response."""
    text = text.strip()
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON. Raw text: {text[:500]}...")
        return None

def analyze_video(file_path: str) -> Optional[Dict[str, Any]]:
    """Uploads, analyzes, and cleans up a single video file."""
    uploaded_file = None
    try:
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
    """Main pipeline for processing one URL."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        return {"error": f"Gemini configuration failed: {e}"}

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
