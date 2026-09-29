"""
NISAR Backend API.
Receives coordinates and dates, searches NISAR scenes on NASA Earthdata.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import asf_search as asf
import traceback

app = FastAPI(title="NISAR Backend")


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NISAR Backend", "version": "1.0"}


@app.post("/search")
def search_scenes(req: AnalyzeRequest):
    """Search NISAR scenes for given coordinates and dates."""
    try:
        point_wkt = f"POINT({req.lon} {req.lat})"
        results = asf.search(
            platform=asf.PLATFORM.NISAR,
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
                "url": p.get("url", ""),
            })

        return {"count": len(scenes), "scenes": scenes}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
