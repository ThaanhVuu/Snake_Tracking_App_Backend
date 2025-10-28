# Hướng dẫn chạy chương trình
- 1. Cài đặt các phụ thuộc
```bash
  pip install -r requirements.txt
```
- 2. Chạy server
```bash
  uvicorn app:app --port 0.0.0.0 --host 8000
```

- 3. Nếu bạn có GPU, bạn có thể cài đặt thêm pytorch cuda để tăng hiệu năng
```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
```