"""
NISAR Backend API.
- /search: find NISAR scenes for a location and date range.
- /analyze: download a GUNW scene, process it, return interferogram PNG.
"""
from fastapi import FastAPI
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

app = FastAPI(title="NISAR Backend")


class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    start_date: str
    end_date: str


@app.get("/")
def read_root():
    return {"status": "ok", "service": "NISAR Backend", "version": "1.1"}


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
    # Try both possible GUNW paths (frequency A and B)
    possible_paths = [
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/unwrappedPhase",
        "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude",
    ]

    with h5py.File(h5path, "r") as f:
        # Find unwrappedPhase dataset
        phase = None
        coh = None
        for key in f.keys():
            pass

        # Walk the tree to find unwrappedPhase
        def find_datasets(name, obj):
            nonlocal phase, coh
            if isinstance(obj, h5py.Dataset):
                if "unwrappedPhase" in name and phase is None:
                    phase = obj[:]
                if "coherenceMagnitude" in name and coh is None:
                    coh = obj[:]
        f.visititems(find_datasets)

        if phase is None:
            raise ValueError("No unwrappedPhase dataset found in file")

        phase = np.array(phase, dtype=np.float32)
        if coh is not None:
            coh = np.array(coh, dtype=np.float32)

    # Convert phase to displacement (L-band, wavelength 24 cm)
    wavelength_cm = 24.0
    disp = phase * wavelength_cm / (4 * np.pi)

    vmin, vmax = np.nanpercentile(disp, [2, 98])

    # Build figure
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


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Download a NISAR GUNW scene and return an interferogram PNG."""
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

        tmpdir = tempfile.mkdtemp()
        print(f"Downloading {scene_name}...")

        # Authenticate if credentials provided in env
        user = os.getenv("EARTHDATA_USERNAME")
        pwd = os.getenv("EARTHDATA_PASSWORD")
        if user and pwd:
            session = asf.ASFSession().auth_with_creds(user, pwd)
        else:
            session = asf.ASFSession()

        files = scene.download(path=tmpdir, session=session)
        if not files:
            return {"error": "Download failed"}

        h5path = files[0]
        outpath = os.path.join(tmpdir, "interferogram.png")

        stats = process_gunw(h5path, outpath)

        return FileResponse(outpath, media_type="image/png",
                            headers={
                                "X-Scene-Name": scene_name,
                                "X-Phase-Min": str(stats["phase_min"]),
                                "X-Phase-Max": str(stats["phase_max"]),
                                "X-Disp-Min-Cm": str(stats["disp_min_cm"]),
                                "X-Disp-Max-Cm": str(stats["disp_max_cm"]),
                            })

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
