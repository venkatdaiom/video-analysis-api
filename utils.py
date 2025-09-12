# utils.py
import json

def flatten_analysis_data(data: dict) -> dict:
    """Flattens the nested JSON structure into a single-level dictionary."""
    flattened = {}
    if "metadata" in data and data["metadata"] is not None:
        for k, v in data["metadata"].items():
            flattened[f"metadata_{k}"] = v
    flattened["analysis_status"] = data.get("analysis_status")
    flattened["error"] = data.get("error")
    if "extracted_features" in data and data["extracted_features"] is not None:
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
