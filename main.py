# main.py
import uuid, pandas as pd, io
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Dict, Any
from analyzer import process_single_video_url, URL_COLUMN_NAME
from utils import flatten_analysis_data

app = FastAPI(title="Batch Video Analysis API", version="2.0.0")

# WARNING: In-memory storage is not suitable for production.
# Task state will be lost if the server restarts. It's OK for Render's free tier.
tasks: Dict[str, Dict[str, Any]] = {}

def run_batch_analysis(batch_id: str, file_contents: bytes, filename: str):
    """Background worker to process the entire uploaded file."""
    try:
        df = pd.read_csv(io.BytesIO(file_contents)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_contents))
        if URL_COLUMN_NAME not in df.columns:
            raise ValueError(f"File must contain a column named '{URL_COLUMN_NAME}'.")
        
        urls = df[URL_COLUMN_NAME].dropna().tolist()
        total = len(urls)
        tasks[batch_id].update({"status": "processing", "total_urls": total, "processed_count": 0, "results": [], "progress": 0.0})

        for i, url in enumerate(urls):
            result = process_single_video_url(url)
            tasks[batch_id]["results"].append(flatten_analysis_data(result))
            tasks[batch_id]["processed_count"] = i + 1
            tasks[batch_id]["progress"] = round(((i + 1) / total) * 100, 2)
        
        tasks[batch_id]["status"] = "completed"
    except Exception as e:
        tasks[batch_id]["status"] = "failed"
        tasks[batch_id]["error"] = str(e)

@app.post("/analyze-batch", status_code=202)
async def start_batch_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a CSV/XLSX file, starts analysis, and returns a batch_id."""
    batch_id = str(uuid.uuid4())
    contents = await file.read()
    tasks[batch_id] = {"status": "accepted"}
    background_tasks.add_task(run_batch_analysis, batch_id, contents, file.filename)
    return {"batch_id": batch_id, "message": "Batch analysis started."}

@app.get("/batch-status/{batch_id}")
def get_batch_status(batch_id: str):
    """Retrieves the status and progress of a batch analysis task."""
    task = tasks.get(batch_id)
    if not task:
        raise HTTPException(status_code=404, detail="Batch ID not found")
    return {k: v for k, v in task.items() if k != 'results'} # Exclude full results from status

@app.get("/results/{batch_id}")
async def get_batch_results(batch_id: str):
    """Returns the final results of a completed batch job as a downloadable CSV."""
    task = tasks.get(batch_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job not complete or found. Status: {task.get('status')}")
    
    results_df = pd.DataFrame(task["results"])
    stream = io.StringIO()
    results_df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=analysis_results_{batch_id}.csv"
    return response
