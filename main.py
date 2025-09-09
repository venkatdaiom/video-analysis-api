# main.py
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any

# Import your custom modules
from analyzer import process_single_video_url
from utils import flatten_analysis_data

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Analysis API",
    description="An API to analyze sales call videos using Google Gemini.",
    version="1.0.0"
)

# --- In-memory storage for task status ---
# In a production system, you'd use a database or Redis for this.
tasks: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for Request/Response ---
class AnalysisRequest(BaseModel):
    video_url: str # Pydantic will validate this is a string, for HttpUrl use HttpUrl

class TaskResponse(BaseModel):
    task_id: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any] | None = None
    error: str | None = None

# --- Background Task Worker ---
def run_analysis(task_id: str, url: str):
    """The function that will run in the background."""
    try:
        # 1. Get the raw JSON analysis
        raw_result = process_single_video_url(url)
        
        # 2. Check for errors during processing
        if raw_result.get("error"):
            tasks[task_id] = {"status": "failed", "error": raw_result["error"]}
            return

        # 3. Flatten the JSON result for easy consumption
        flattened_result = flatten_analysis_data(raw_result)
        
        # 4. Store the final result
        tasks[task_id] = {"status": "completed", "result": flattened_result}

    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": f"An unexpected error occurred: {str(e)}"}


# --- API Endpoints ---
@app.post("/analyze", response_model=TaskResponse, status_code=202)
def start_video_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Accepts a video URL and starts the analysis in the background.
    Returns a task ID to check the status later.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}
    
    # Add the long-running analysis function to the background
    background_tasks.add_task(run_analysis, task_id, request.video_url)
    
    return {"task_id": task_id, "message": "Analysis has been started."}


@app.get("/status/{task_id}", response_model=StatusResponse)
def get_analysis_status(task_id: str):
    """
    Retrieves the status and result of an analysis task using its ID.
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response_data = {
        "task_id": task_id,
        "status": task.get("status"),
        "result": task.get("result"),
        "error": task.get("error")
    }
    return response_data
