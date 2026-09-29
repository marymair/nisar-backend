"""
NISAR Backend API v2.
- /search: find NISAR scenes for a location and date range.
- /analyze: start async download+processing, return job_id.
- /status/{job_id}: check job progress.
- /image/{job_id}: get resulting PNG.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
import requests

app = FastAPI(title="NISAR Backend")

# In-memory job store: job_id -> {status, image_path, error, scene_name}
JOBS = {}


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NISAR Backend", "version": "2.0"}


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


def process_gunw(h5path: str, outpath: str) -> dict:
    """Open a GUNW HDF5 file and build the interferogram figure."""
    phase = None
    coh = None

    with h5py.File(h5path, "r") as f:
        def find_datasets(name, obj):
            nonlocal phase, coh
            if isinstance(obj, h5py.Dataset):
                if "unwrappedPhase" in name and phase is None:
                    phase = obj[:]
                    print(f"[process] Found phase dataset: {name} shape={obj.shape}", flush=True)
                if "coherenceMagnitude" in name and coh is None:
                    coh = obj[:]
                    print(f"[process] Found coherence dataset: {name} shape={obj.shape}", flush=True)
        f.visititems(find_datasets)

    if phase is None:
        raise ValueError("No unwrappedPhase dataset found in file")

    phase = np.array(phase, dtype=np.float32)
    if coh is not None:
        coh = np.array(coh, dtype=np.float32)

    wavelength_cm = 24.0
    disp = phase * wavelength_cm / (4 * np.pi)

    vmin, vmax = np.nanpercentile(disp, [2, 98])

    if coh is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
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

    plt.suptitle("NISAR GUNW Interferogram", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "phase_min": float(np.nanmin(phase)),
        "phase_max": float(np.nanmax(phase)),
        "disp_min_cm": float(np.nanmin(disp)),
        "disp_max_cm": float(np.nanmax(disp)),
    }


def run_analysis(job_id: str, scene, tmpdir: str, user: str, pwd: str):
    """Background worker: download + process + save PNG."""
    try:
        scene_name = scene.properties.get("sceneName", "scene")
        url = scene.properties.get("url", "")
        print(f"[job {job_id}] Starting: {scene_name}", flush=True)
        print(f"[job {job_id}] URL: {url}", flush=True)

        if not url:
            raise ValueError("Scene has no URL in properties")

        # Download directly with requests + auth
        h5path = os.path.join(tmpdir, scene_name + ".h5")
        print(f"[job {job_id}] Downloading via requests...", flush=True)

        with requests.get(url, auth=(user, pwd), stream=True, timeout=600) as r:
            print(f"[job {job_id}] HTTP status: {r.status_code}", flush=True)
            r.raise_for_status()
            total = 0
            with open(h5path, "wb") as fout:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fout.write(chunk)
                        total += len(chunk)
                        if total % (50 * 1024 * 1024) < (1024 * 1024):
                            print(f"[job {job_id}] Downloaded {total / 1e6:.1f} MB", flush=True)

        print(f"[job {job_id}] Download complete: {total / 1e6:.1f} MB", flush=True)

        # Process
        outpath = os.path.join(tmpdir, "interferogram.png")
        stats = process_gunw(h5path, outpath)
        print(f"[job {job_id}] Processing done: {stats}", flush=True)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["image_path"] = outpath
        JOBS[job_id]["scene_name"] = scene_name
        JOBS[job_id]["stats"] = stats

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[job {job_id}] ERROR: {err}", flush=True)
        print(f"[job {job_id}] TRACE: {traceback.format_exc()}", flush=True)
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = err


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Start async analysis. Returns job_id immediately."""
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
            return {"error": "No GUNW scenes found for this location and period"}

        scene = results[0]
        scene_name = scene.properties.get("sceneName", "scene")

        user = os.getenv("EARTHDATA_USERNAME")
        pwd = os.getenv("EARTHDATA_PASSWORD")
        if not user or not pwd:
            return {"error": "EARTHDATA_USERNAME / EARTHDATA_PASSWORD not set"}

        tmpdir = tempfile.mkdtemp()
        job_id = str(uuid.uuid4())[:8]

        JOBS[job_id] = {
            "status": "processing",
            "scene_name": scene_name,
            "image_path": None,
            "error": None,
        }

        thread = threading.Thread(
            target=run_analysis,
            args=(job_id, scene, tmpdir, user, pwd),
            daemon=True,
        )
        thread.start()

        return {
            "job_id": job_id,
            "status": "processing",
            "scene_name": scene_name,
            "message": "Analysis started. Poll /status/{job_id}.",
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    """Check status of an analysis job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "scene_name": job.get("scene_name"),
        "error": job.get("error"),
        "stats": job.get("stats"),
    }


@app.get("/image/{job_id}")
def job_image(job_id: str):
    """Return the resulting PNG."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    return FileResponse(job["image_path"], media_type="image/png")
