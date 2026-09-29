"""
NISAR Backend API.
Receives coordinates and dates, searches NISAR scenes on NASA Earthdata,
downloads data, processes it, and returns the interferogram.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asf_search as asf
import os
import tempfile
import traceback

app = FastAPI(title="NISAR Backend")


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    start_date: str   # "2026-09-01"
    end_date: str     # "2026-09-28"


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NISAR Backend", "version": "1.0"}


@app.post("/search")
def search_scenes(req: AnalyzeRequest):
    """Search NISAR scenes for given coordinates and dates."""
    try:
        point_wkt = f"POINT({req.lon} {req.lat})"
        results = asf.search(
            dataset="NISAR",
            intersectsWith=point_wkt,
            start=f"{req.start_date}T00:00:00Z",
            end=f"{req.end_date}T23:59:59Z",
            maxResults=10,
        )

        scenes = []
        for scene in results:
            p = scene.properties
            scenes.append({
                "name": p.get("sceneName", "unknown"),
                "date": p.get("startTime", "unknown"),
                "level": p.get("processingLevel", "unknown"),
                "size_mb": p.get("bytes", 0) / 1e6 if isinstance(p.get("bytes"), (int, float)) else None,
                "url": p.get("url", ""),
            })

        return {"count": len(scenes), "scenes": scenes}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
