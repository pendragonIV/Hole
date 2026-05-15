# Colab — các bước tiếp theo (sau `git clone`)

Chạy **theo thứ tự** từng ô trong Colab. Giả định bạn đã clone [pendragonIV/Hole](https://github.com/pendragonIV/Hole) và `pwd` là **root repo** (cùng cấp với `README.md`, `docs/`, thư mục ẩn `.git`).

Nếu đang bị lồng `Hole/Hole`, xem lại phần “Chuẩn hoá thư mục” trong [`LO_TRINH_QWEN_BASE.md`](./LO_TRINH_QWEN_BASE.md) mục 2 hoặc README repo — hoặc chạy:

```python
%cd /content
!rm -rf Hole
!git clone https://github.com/pendragonIV/Hole.git
%cd Hole
```

---

## Bước 1 — Xác nhận GPU

```python
!nvidia-smi
```

**Mục đích:** Biết GPU + VRAM để chọn cỡ model (xem [`QUYET_DINH_VA_THUC_HIEN.md`](./QUYET_DINH_VA_THUC_HIEN.md): mặc định `Qwen2.5-Coder-1.5B-Instruct` cho T4).

---

## Bước 2 — Google Drive (nên bật trước khi tải model / lưu checkpoint)

```python
from google.colab import drive
drive.mount("/content/drive")

OUTPUT_DIR = "/content/drive/MyDrive/qwen-ft/run1"  # đổi tên nếu muốn
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Mục đích:** Colab ngắt phiên; adapter/weight lưu Drive không mất khi disconnect.

*(Trên Colab, biến `OUTPUT_DIR` kiểu này chỉ tồn tại trong session — ghi nhớ path hoặc dán vào notebook Unsloth sau.)*

---

## Bước 3 — Token Hugging Face (nếu cần tải model/dataset private)

1. Colab: **Secrets** (khóa) → thêm secret tên `HF_TOKEN` (giá trị = token từ [HF Settings](https://huggingface.co/settings/tokens)).

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**Mục đích:** Tránh lỗi 401 khi `from_pretrained` với repo gated hoặc rate limit.

---

## Bước 4 — Notebook train thật: dùng Unsloth (không train trong repo Hole)

Repo **Hole** chỉ chứa **tài liệu lộ trình**. Train fine-tune thực tế nên bắt đầu từ notebook **chính thức** của [unslothai/unsloth](https://github.com/unslothai/unsloth) (tìm Colab **Qwen2.5-Coder** / Qwen Coder SFT).

**Việc bạn làm:**

1. Mở notebook Colab từ link trong repo Unsloth (hoặc tìm `Unsloth Qwen Coder fine tuning Colab`).
2. **File → Save a copy in Drive** để có bản của riêng bạn.
3. Trong notebook đó, chạy lần lượt các ô: cài pip (theo **đúng** block của họ — có thể có `unsloth[colab-new]`).
4. **Runtime → Restart runtime** nếu notebook yêu cầu sau pip.
5. Chỉnh biến tải model thành (vòng 1 an toàn VRAM):

   ```python
   model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
   ```

   (Lý do: [`QUYET_DINH_VA_THUC_HIEN.md`](./QUYET_DINH_VA_THUC_HIEN.md).)

6. Trỏ `output_dir` / lưu adapter vào **`OUTPUT_DIR`** trên Drive (Bước 2).

**Mục đích:** Tránh tự viết train loop; luôn dùng bản Unsloth mới nhất cho Colab/CUDA hiện tại.

---

## Bước 5 — Dataset

- Lần đầu: dùng **dataset mẫu** trong notebook Unsloth để chạy thông end-to-end.
- Sau đó: thay bằng `train.jsonl` của bạn (upload hoặc đặt trên Drive), format đúng cột mà notebook đang map (instruction / messages).

**Mục đích:** Ổn định pipeline trước khi đổ data riêng.

---

## Bước 6 — Chạy train thử (ngắn)

- Giảm epoch / max_steps cho **smoke test** (~5–15 phút).
- Kiểm tra loss in ra; thư mục Drive có file adapter / checkpoint.

**Mục đích:** Xác nhận không OOM, không lỗi template tokenizer.

---

## Bước 7 — Sau khi train ổn

- Đọc tiếp [`LO_TRINH_QWEN_BASE.md`](./LO_TRINH_QWEN_BASE.md) mục **5–7** (eval, GGUF, Ollama / VS Code).
- Cập nhật tài liệu trong repo Hole (markdown) nếu bạn muốn ghi chú hyperparam riêng — rồi `git add` / `commit` / `push` từ **máy local** hoặc cấu hình git + token trên Colab (phức tạp hơn; khuyến nghị chỉnh doc trên PC rồi push).

---

## Tóm tắt một dòng

**Sau clone Hole:** kiểm GPU → mount Drive → HF token → mở **notebook Unsloth** trên Drive → đặt `Qwen2.5-Coder-1.5B-Instruct` → train thử ngắn → lưu ra Drive.
