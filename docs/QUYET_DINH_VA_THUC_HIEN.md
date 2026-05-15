# Quyết định lộ trình & phần đã thực hiện thay bạn

Tài liệu này ghi **các lựa chọn cụ thể**, **lý do**, và phân biệt rõ việc đã làm được **trong môi trường dev (tra cứu + thiết kế)** vs việc **bạn phải tự làm trên trình duyệt (Google Colab)**.

---

## A. Giới hạn thực tế (trung thực)

| Việc | Trạng thái |
|------|------------|
| Tra cứu repo **QwenLM**, model card **Hugging Face**, độ lớn model | **Đã làm** (công khai trên web) |
| Chốt ID model + stack train + nguyên tắc Colab | **Đã làm** — ghi dưới đây |
| Mở phiên Colab, bấm Play, tốn GPU Google, tải weight vài GB | **Chưa thể thay bạn** — cần tài khoản Google và trình duyệt của bạn |

---

## B. Chọn “nhà” tài liệu trên GitHub: **Qwen3-Coder**

**Đã chọn:** theo dõi và đọc hướng dẫn chính thức tại **[QwenLM/Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)** trong tổ chức **[QwenLM](https://github.com/QwenLM)**.

**Lý do:**

- Đúng **tuyến code** của Alibaba (mô tả org: *code version of Qwen3*), không lẫn với **Qwen3-VL** (ảnh), **Qwen-Image**, **Qwen3-Omni**.
- **Qwen-Agent**, **qwen-code** là **framework / app** (agent terminal), không phải bộ weight nhỏ gọn để bạn nhét vào ô `model_name` fine-tune kiểu học viện — dùng sau khi đã có model.

---

## C. Chọn **Model ID trên Hugging Face** để train trên Colab (T4 ~16GB)

### C.1 Phân tích nhanh Qwen3-Coder trên HF

Các bản **Qwen3-Coder** đang thấy rộng rãi trên Hugging Face thường là **MoE / cỡ rất lớn** (ví dụ [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)) — **không phù hợp** làm mặc định cho **Colab miễn phí + QLoRA** (OOM hoặc cực chật / không ổn định).

### C.2 Quyết định base cho **vòng 1 (Colab T4, 4-bit, Unsloth)**

**Đã chọn (mặc định an toàn):**

- **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** — [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

**Lý do:**

- **Instruct** + dòng **Coder** → sẵn sàng cho SFT / instruction tuning giống Copilot.
- **1.5B** + **4-bit** + LoRA → xác suất chạy trơn trên **T4** cao, ít OOM, vòng train đầu để học **pipeline** (data → train → lưu adapter).
- Apache-2.0 (theo card công khai), ecosystem GGUF/AWQ đã quen thuộc.

**Nâng cấp khi VRAM dư (cùng dòng, vẫn dense, dễ Unsloth):**

| Ưu tiên | Model ID | Ghi chú |
|---------|----------|---------|
| Cân bằng | `Qwen/Qwen2.5-Coder-3B-Instruct` | [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) |
| Mạnh hơn nếu T4/L4 ổn | `Qwen/Qwen2.5-Coder-7B-Instruct` | [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) — có thể cần giảm `max_seq_length` / batch trên T4 |

**Khi nào quay lại Qwen3-Coder trên HF?**

- Khi bạn có **GPU ≥ ~24GB VRAM** hoặc **Colab Pro+ / RunPod** và đã xác nhận notebook Unsloth hỗ trợ đúng kiến trúc (MoE / Next), **sau** khi vòng 1 với 2.5-Coder đã chạy xong.

---

## D. Stack train trên Colab

**Đã chọn:** **Unsloth** + **QLoRA (4-bit)** + **TRL / PEFT** (theo notebook mẫu Unsloth cho Qwen Coder).

**Lý do:**

- Giảm VRAM và thời gian so với full fine-tune; đúng bài toán “cá nhân / Colab”.
- Notebook upstream cập nhật nhanh khi Qwen đổi config — ít tự viết PyTorch từ đầu.

**Điểm xuất phát notebook:** repo **[unslothai/unsloth](https://github.com/unslothai/unsloth)** — tìm Colab **Qwen / Qwen2.5-Coder** fine-tune (từ khóa: `Unsloth Qwen Coder fine tuning Colab`).

---

## E. Các bước Colab — bạn làm trên trình duyệt (checklist)

1. **Runtime → GPU** → chạy `!nvidia-smi` → ghi VRAM.
2. **Mount Drive** → `output_dir` trỏ vào `MyDrive/qwen-ft/`.
3. **HF_TOKEN** (Secrets) nếu cần tải gated / tăng rate limit.
4. Cài pip **theo đúng ô đầu** của notebook Unsloth (luôn ưu tiên bản mới nhất trên repo họ).
5. Trong `from_pretrained`, đặt:

   ```python
   model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
   ```

   (Đổi sang 3B/7B khi đã chạy thử 1.5B thành công.)

6. Train ngắn → lưu adapter lên Drive → eval trên vài prompt tay.

---

## F. Ô lệnh Colab copy-paste (khung tối thiểu)

**Lưu ý:** Dòng cài **Unsloth** có thể thay đổi — nếu lệnh dưới lỗi, thay bằng block pip **trong notebook Unsloth** bạn đang mở.

```python
# 1) GPU
!nvidia-smi
```

```python
# 2) Drive (tùy chọn nhưng nên có)
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_DIR = "/content/drive/MyDrive/qwen-ft/run1"
```

```bash
# 3) Thư viện — bổ sung/chỉnh theo notebook Unsloth hiện tại
!pip install -U pip
!pip install --upgrade transformers peft trl accelerate bitsandbytes
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

```python
# 4) Token HF (Colab Secrets: tên biến HF_TOKEN)
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
```

Phần **load model + SFTTrainer** — giữ nguyên từ notebook Unsloth; chỉ sửa `model_name` và đường dẫn dataset/output như mục E.

---

## G. Liên kết tài liệu trong repo

- Lộ trình tổng quát + Colab chi tiết: [`LO_TRINH_QWEN_BASE.md`](./LO_TRINH_QWEN_BASE.md)
- Bản quyết định này: [`QUYET_DINH_VA_THUC_HIEN.md`](./QUYET_DINH_VA_THUC_HIEN.md)

---

*Tóm tắt một dòng: **Tài liệu & stack = QwenLM + Unsloth; base train Colab mặc định = Qwen2.5-Coder-1.5B-Instruct** vì cân VRAM; **Qwen3-Coder** trên HF chủ yếu cỡ lớn/MoE — để giai đoạn sau hoặc GPU mạnh hơn.*
