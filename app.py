import shutil
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

# ====== Cấu hình ======
MODEL_PATH = "best.pt"                 # path tới model của bạn
TRACKER_CFG = "tracker_bytetrack.yaml" # ByteTrack config
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ====== Khởi tạo ứng dụng ======
app = FastAPI(title="Snake Tracking Backend", version="1.0")

# ====== Cấu hình CORS ======
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # cho phép React FE
    allow_credentials=True,
    allow_methods=["*"],  # cho phép tất cả method (GET, POST, ...)
    allow_headers=["*"]  # cho phép tất cả header
)

# ====== Load model 1 lần khi start server ======
if not Path(MODEL_PATH).exists():
    raise RuntimeError(f"Không thấy model: {MODEL_PATH}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)

def _safe_filename(stem: str, suffix: str) -> Path:
    return OUTPUT_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

def _cv2_imdecode_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được ảnh (định dạng không hợp lệ?)")
    return img

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "torch_cuda": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

# =============== ẢNH: detect (1 frame) =================
@app.post("/track/image")
async def track_image(file: UploadFile = File(...), conf: Optional[float] = 0.25, iou: Optional[float] = 0.45):
    """
    Nhận 1 ảnh, vẽ bbox + nhãn + xác suất. (Ảnh đơn thì không cần tracker)
    Trả về ảnh đã annotate (PNG).
    """
    try:
        raw = await file.read()
        img = _cv2_imdecode_to_bgr(raw)

        results = model.predict(
            img, conf=conf, iou=iou, device=device, verbose=False
        )
        # Vẽ kết quả lên ảnh
        plotted = results[0].plot()  # BGR ndarray đã vẽ bbox

        out_path = _safe_filename("image_tracked", ".png")
        cv2.imwrite(str(out_path), plotted)

        return FileResponse(path=str(out_path), media_type="image/png", filename=out_path.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =============== VIDEO: track (nhiều frame) =================
# noinspection PyTypeChecker,PyBroadException
@app.post("/track/video")
async def track_video(file: UploadFile = File(...), conf: Optional[float] = 0.25, iou: Optional[float] = 0.45):
    """
    Nhận video, chạy YOLO + ByteTrack để gán ID ổn định theo thời gian.
    Trả về video đã annotate (MP4).
    """
    # Lưu tạm input
    try:
        suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
        temp_in = _safe_filename("input", suffix)
        with open(temp_in, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi lưu tệp tạm: {e}")

    try:
        # Ultralytics sẽ tự tạo file output trong runs/track/... Chúng ta sẽ copy ra outputs/
        results = model.track(
            source=str(temp_in),
            tracker=TRACKER_CFG,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            stream=False,
            save=True,          # rất quan trọng: ghi video đã annotate
            exist_ok=True
        )

        # Tìm file video kết quả mà ultralytics vừa tạo (results có .save_dir và .path)
        # results có thể là list; lấy phần tử đầu
        res0 = results[0] if isinstance(results, list) else results
        # save_dir kiểu .../runs/track/expN
        save_dir = Path(res0.save_dir)
        # Tìm file .mp4 trong thư mục đó
        mp4s = list(save_dir.glob("*.mp4"))
        if not mp4s:
            # fallback: một số bản sẽ xuất vào 'tracks' subdir
            mp4s = list((save_dir / "tracks").glob("*.mp4"))

        if not mp4s:
            raise RuntimeError("Không tìm thấy video output sau khi track.")

        final_out = _safe_filename("video_tracked", ".mp4")
        shutil.copy2(mp4s[0], final_out)

        return FileResponse(path=str(final_out), media_type="video/mp4", filename=final_out.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý video: {e}")
    finally:
        # Dọn file input tạm
        try:
            if temp_in.exists():
                temp_in.unlink()
        except Exception:
            pass

# =============== Endpoint tiện ích: thông tin model ================
@app.get("/model/info")
def model_info():
    try:
        names = model.names
        return JSONResponse({
            "model": MODEL_PATH,
            "device": device,
            "classes": names,
            "num_classes": len(names)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
