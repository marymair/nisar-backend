"""
NISAR Backend API v2.5
- /search: find NISAR scenes
- /check_regions: batch check availability
- /analyze: async download+process
- /status, /image
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import asf_search as asf
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import tempfile
import traceback
import threading
import uuid

app = FastAPI(title="NISAR Backend")
JOBS = {}


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str


class RegionCheckRequest(BaseModel):
    regions: List[dict]
    start_date: str
    end_date: str


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NISAR Backend", "version": "2.5"}


@app.post("/search")
def search_scenes(req: AnalyzeRequest):
    try:
        point_wkt = f"POINT({req.lon} {req.lat})"
        results = asf.search(
            platform=asf.PLATFORM.NISAR,
            intersectsWith=point_wkt,
            start=f"{req.start_date}T00:00:00Z",
            end=f"{req.end_date}T23:59:59Z",
            maxResults=20,
        )
        scenes = []
        for scene in results:
            p = scene.properties
            # Extract scene center coords from metadata if available
            lat = p.get("centerLat", req.lat) or req.lat
            lon = p.get("centerLon", req.lon) or req.lon
            scenes.append({
                "name": p.get("sceneName", "unknown"),
                "date": p.get("startTime", "unknown"),
                "level": p.get("processingLevel", "unknown"),
                "url": p.get("url", ""),
                "lat": lat,
                "lon": lon,
            })
        # Sort: GUNW first, then by date
        order = {"GUNW": 0, "RUNW": 1, "GSLC": 2, "GCOV": 3, "RRSD": 4, "SME2": 5}
        scenes.sort(key=lambda s: (order.get(s.get("level", ""), 99), s.get("date", "")))
        return {"count": len(scenes), "scenes": scenes}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.post("/check_regions")
def check_regions(req: RegionCheckRequest):
    results = []
    for region in req.regions:
        try:
            point_wkt = f"POINT({region['lon']} {region['lat']})"
            scenes = asf.search(
                platform=asf.PLATFORM.NISAR,
                intersectsWith=point_wkt,
                start=f"{req.start_date}T00:00:00Z",
                end=f"{req.end_date}T23:59:59Z",
                maxResults=5,
            )
            results.append({
                "name": region.get("name", "unknown"),
                "lat": region["lat"],
                "lon": region["lon"],
                "count": len(scenes),
                "available": len(scenes) > 0,
                "levels": list({s.properties.get("processingLevel", "?") for s in scenes}),
            })
        except Exception as e:
            results.append({
                "name": region.get("name", "unknown"),
                "lat": region["lat"],
                "lon": region["lon"],
                "count": 0,
                "available": False,
                "error": str(e),
            })
    return {"results": results}


def process_gunw(h5path: str, outpath: str) -> dict:
    phase = None
    coh = None
    with h5py.File(h5path, "r") as f:
        def find_datasets(name, obj):
            nonlocal phase, coh
            if isinstance(obj, h5py.Dataset):
                if "unwrappedPhase" in name and phase is None:
                    phase = obj[:]
                if "coherenceMagnitude" in name and coh is None:
                    coh = obj[:]
        f.visititems(find_datasets)

    if phase is None:
        raise ValueError("No unwrappedPhase dataset found")

    STEP = 3
    phase = np.array(phase[::STEP, ::STEP], dtype=np.float32)
    if coh is not None:
        coh = np.array(coh[::STEP, ::STEP], dtype=np.float32)

    wavelength_cm = 24.0
    disp = phase * wavelength_cm / (4 * np.pi)
    vmin, vmax = np.nanpercentile(disp, [2, 98])

    if coh is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = list(axes)

    ax_idx = 0
    if coh is not None:
        im0 = axes[ax_idx].imshow(coh, cmap="gray", vmin=0, vmax=1)
        axes[ax_idx].set_title("Coherence (signal reliability)")
        axes[ax_idx].axis("off")
        plt.colorbar(im0, ax=axes[ax_idx], fraction=0.046)
        ax_idx += 1

    im1 = axes[ax_idx].imshow(phase, cmap="twilight",
                              vmin=np.nanpercentile(phase, 2),
                              vmax=np.nanpercentile(phase, 98))
    axes[ax_idx].set_title("Unwrapped Phase (radians)")
    axes[ax_idx].axis("off")
    plt.colorbar(im1, ax=axes[ax_idx], fraction=0.046)
    ax_idx += 1

    im2 = axes[ax_idx].imshow(disp, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[ax_idx].set_title("Surface Displacement (cm)")
    axes[ax_idx].axis("off")
    cb = plt.colorbar(im2, ax=axes[ax_idx], fraction=0.046)
    cb.set_label("displacement, cm")

    plt.suptitle("NISAR GUNW Interferogram", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "phase_min": float(np.nanmin(phase)),
        "phase_max": float(np.nanmax(phase)),
        "disp_min_cm": float(np.nanmin(disp)),
        "disp_max_cm": float(np.nanmax(disp)),
    }


def run_analysis(job_id, scene, tmpdir, user, pwd):
    try:
        scene_name = scene.properties.get("sceneName", "scene")
        print(f"[job {job_id}] Start: {scene_name}", flush=True)
        session = asf.ASFSession()
        session.auth_with_creds(user, pwd)
        print(f"[job {job_id}] auth OK", flush=True)
        scene.download(path=tmpdir, session=session)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".h5")]
        print(f"[job {job_id}] files: {files}", flush=True)
        if not files:
            raise RuntimeError("No .h5 downloaded")
        h5path = os.path.join(tmpdir, files[0])
        outpath = os.path.join(tmpdir, "interferogram.png")
        stats = process_gunw(h5path, outpath)
        print(f"[job {job_id}] done: {stats}", flush=True)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["image_path"] = outpath
        JOBS[job_id]["stats"] = stats
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[job {job_id}] ERROR: {err}", flush=True)
        print(traceback.format_exc(), flush=True)
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = err


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        point_wkt = f"POINT({req.lon} {req.lat})"
        results = asf.search(
            platform=asf.PLATFORM.NISAR,
            intersectsWith=point_wkt,
            start=f"{req.start_date}T00:00:00Z",
            end=f"{req.end_date}T23:59:59Z",
            processingLevel="GUNW",
            maxResults=1,
        )
        if not results:
            return {"error": "No GUNW scenes for this location and period"}
        scene = results[0]
        user = os.getenv("EARTHDATA_USERNAME")
        pwd = os.getenv("EARTHDATA_PASSWORD")
        if not user or not pwd:
            return {"error": "EARTHDATA creds not set"}
        tmpdir = tempfile.mkdtemp()
        job_id = str(uuid.uuid4())[:8]
        JOBS[job_id] = {"status": "processing", "image_path": None, "error": None}
        threading.Thread(target=run_analysis, args=(job_id, scene, tmpdir, user, pwd), daemon=True).start()
        return {"job_id": job_id, "status": "processing", "scene_name": scene.properties.get("sceneName", "scene")}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {"job_id": job_id, "status": job["status"], "error": job.get("error"), "stats": job.get("stats")}


@app.get("/image/{job_id}")
def job_image(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    return FileResponse(job["image_path"], media_type="image/png")
