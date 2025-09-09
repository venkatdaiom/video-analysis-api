# utils.py
import json

def flatten_analysis_data(data: dict) -> dict:
    """
    Flattens the nested JSON structure of the video/call analysis into a single-level dictionary.
    """
    flattened = {}

    # --- Metadata ---
    if "metadata" in data and data["metadata"] is not None:
        for k, v in data["metadata"].items():
            flattened[f"metadata_{k}"] = v

    flattened["analysis_status"] = data.get("analysis_status")
    flattened["error"] = data.get("error")

    if "extracted_features" in data and data["extracted_features"] is not None:
        extracted = data["extracted_features"]

        if "visual_analysis" in extracted and extracted.get("visual_analysis") is not None:
            for k, v in extracted["visual_analysis"].items():
                flattened[f"visual_analysis_{k}"] = v

        if "product_demo_analysis" in extracted and extracted.get("product_demo_analysis") is not None:
            for k, v in extracted["product_demo_analysis"].items():
                flattened[f"product_demo_analysis_{k}"] = v

        if "conversation_analysis" in extracted and extracted.get("conversation_analysis") is not None:
            for k, v in extracted["conversation_analysis"].items():
                if isinstance(v, list):
                    flattened[f"conversation_analysis_{k}"] = "; ".join(map(str, v))
                else:
                    flattened[f"conversation_analysis_{k}"] = v

        if "relax_framework" in extracted and extracted.get("relax_framework") is not None:
            for stage, details in extracted["relax_framework"].items():
                if isinstance(details, dict):
                    for k, v in details.items():
                        flattened[f"relax_framework_{stage}_{k}"] = v
                else:
                    flattened[f"relax_framework_{stage}"] = details

        if "customer_signals" in extracted and extracted.get("customer_signals") is not None:
            for k, v in extracted["customer_signals"].items():
                if isinstance(v, list):
                    flattened[f"customer_signals_{k}"] = "; ".join(map(str, v))
                else:
                    flattened[f"customer_signals_{k}"] = v

    return flattened
