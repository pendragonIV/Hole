# Lộ trình fine-tune Coding Model — Base Qwen / Qwen-Coder

Tài liệu này mô tả các bước thực hiện khi chọn **họ Qwen (Alibaba)** làm mô hình nền trên Hugging Face, dùng **QLoRA** (ví dụ qua **Unsloth**), xuất **GGUF**, chạy local (**Ollama** / **LM Studio**) và gắn **VS Code** (Continue / Twinny).

**Huấn luyện (train):** ưu tiên mô tả cho **Google Colab** (GPU miễn phí hoặc Colab Pro); phần cuối vẫn dùng máy cá nhân chỉ để **chạy model** sau khi có GGUF.

**Quyết định đã chốt + lý do (cập nhật theo tra cứu công khai):** xem [`QUYET_DINH_VA_THUC_HIEN.md`](./QUYET_DINH_VA_THUC_HIEN.md) — trong đó có **Model ID mặc định cho Colab T4** và giải thích vì sao ưu tiên **Qwen2.5-Coder** cho vòng 1 thay vì các bản **Qwen3-Coder** cỡ lớn trên HF.

> **Lưu ý:** ID model trên Hugging Face thay đổi theo thời gian. Trước mỗi phiên làm việc, hãy mở [Hugging Face — Qwen](https://huggingface.co/Qwen) và chọn đúng repo **Instruct** hoặc **Coder** phù hợp kích thước + license bạn chấp nhận.

---

## 0. Chuẩn bị & phạm vi

| Hạng mục | Gợi ý |
|----------|--------|
| Mục tiêu | Model viết/sửa/giải thích code theo **stack + convention** của bạn |
| Không làm | Pre-training từ số không (Terabyte, hàng nghìn GPU) |
| Phần cứng (train) | **Colab:** GPU NVIDIA (thường T4 ~16GB VRAM free; L4/A100 tuỳ gói). VRAM thấp → model nhỏ hơn + `max_seq_length` thấp + **luôn 4-bit** |
| Dữ liệu | JSONL dạng instruction (instruction / input / output hoặc tương đương chat) |

**Chọn size Qwen (tham số) theo VRAM (tham khảo):**

| VRAM (gần đúng) | Hướng chọn base |
|-----------------|-----------------|
| ~8 GB | 0.5B–1.5B–3B (Coder nếu có), giảm `max_seq_length` |
| 12–16 GB | 3B–7B / 8B ở 4-bit, QLoRA |
| 24 GB+ | 7B–8B thoải mái hơn; 14B cần cân nhắc kỹ |

---

## 1. Chọn đúng base Qwen trên Hugging Face

1. Vào [huggingface.co/Qwen](https://huggingface.co/Qwen) (hoặc tìm `Qwen2.5-Coder`, `Qwen3`, v.v.).
2. Ưu tiên bản **Coder** hoặc **Instruct** (đã align chat / code), không dùng nhầm bản **base** thuần nếu notebook của bạn giả định chat template instruct.
3. Ghi lại:
   - **Model ID** đầy đủ (ví dụ dạng `Qwen/...`).
   - **Context length** tối đa model hỗ trợ vs **max_seq_length** bạn dùng khi train (không vượt quá khả năng + VRAM).
4. Kiểm tra **license** và điều khoản thương mại nếu dùng cho công ty.

**Checkpoint:** Bạn có 1 dòng `model_name = "Qwen/..."` chính xác + hiểu bản đó là Instruct hay Base.

---

## 2. Huấn luyện trên Google Colab (chi tiết)

### 2.1 Tạo notebook & bật GPU

1. Vào [Google Colab](https://colab.research.google.com/) (tài khoản Google).
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** (hoặc GPU cao hơn nếu Colab Pro / Pro+).
3. Xác nhận GPU: ô đầu tiên chạy:

```python
!nvidia-smi
```

Ghi lại **tên GPU** và dung lượng **VRAM** để chọn đúng cỡ Qwen (mục 0).

### 2.2 Notebook mẫu Unsloth + Qwen (khuyến nghị)

- Trên GitHub [unslothai/unsloth](https://github.com/unslothai/unsloth), tìm notebook **Colab** cho **Qwen / Qwen2.5-Coder** (fine-tune / SFT). Đó là điểm xuất phát nhanh nhất: đã có sẵn tải model, dataset mẫu, `SFTTrainer`, QLoRA.
- Tìm kiếm gợi ý: `Unsloth Qwen Coder fine tuning Colab` — mở notebook chính thức từ repo Unsloth rồi **Save a copy in Drive** để chỉnh `model_name` và data của bạn.

### 2.3 Gắn Google Drive (lưu adapter / checkpoint — nên làm)

Phiên Colab **có thể ngắt** sau một lúc; model lớn không nên chỉ lưu trong RAM/disk tạm.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Đặt thư mục làm việc, ví dụ: `/content/drive/MyDrive/qwen-ft/` — trong notebook, mọi `output_dir` / `save_pretrained` trỏ vào đây.

### 2.4 Hugging Face (tải model & dataset private)

Nếu model/dataset cần đăng nhập:

1. Tạo token: [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Colab: **Secrets** (biểu tượng khóa) → thêm `HF_TOKEN` (hoặc dùng `huggingface-cli login` trong notebook — kém an toàn hơn nếu chia sẻ notebook).

Ví dụ dùng biến môi trường (chỉnh theo cách Colab của bạn đọc secret):

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

### 2.5 Cài thư viện (ô đầu notebook — cập nhật theo doc Unsloth)

Luôn ưu tiên **đoạn pip trong notebook Unsloth** bạn đang mở (họ cập nhật theo Colab). Khung tổng quát:

```bash
!pip install -U pip
!pip install --upgrade transformers peft trl accelerate bitsandbytes
# Unsloth: dùng đúng dòng install trên README Unsloth cho Colab, ví dụ:
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

- **Qwen3 / kiến trúc mới:** nếu báo lỗi load config, tăng phiên bản `transformers` và cài lại Unsloth từ git như hướng dẫn mới nhất trên repo.
- Sau cài: **Runtime → Restart runtime** nếu notebook yêu cầu.

### 2.6 Đưa dataset của bạn vào Colab

| Cách | Khi nào dùng |
|------|----------------|
| **Upload** file `.jsonl` nhỏ | Thử nhanh vài MB |
| **Google Drive** | Dataset lớn; đọc path `/content/drive/MyDrive/...` |
| **`datasets.load_dataset`** từ Hugging Face | Dùng bộ public có sẵn |

Đảm bảo **cột / format** khớp với hàm format trong notebook Unsloth (instruction vs chat).

### 2.7 Giới hạn Colab & cách tránh mất công

| Hạn chế | Cách xử lý |
|---------|------------|
| Phiên bị **ngắt** (idle / hết slot) | Lưu adapter mỗi N bước vào Drive; train nhiều epoch ngắn thay vì một block cực dài |
| **OOM** T4 | Giảm `max_seq_length`, batch, chọn Qwen **1.5B / 3B**; giữ `load_in_4bit=True` |
| **Timeout** một ô chạy quá lâu | Chia nhỏ bước; hoặc Colab Pro cho session dài hơn |
| Mất kết nối khi download model | Bật cache HF trên Drive (`HF_HOME` trỏ vào thư mục Drive) để lần sau không tải lại từ đầu |

Gợi ý cache trên Drive (tùy chọn, tốn dung lượng Drive):

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### 2.8 Sau khi train trên Colab

- File thường có: **adapter LoRA** (nhẹ) và/hoặc **merged weights** — tùy notebook.
- **GGUF:** nhiều người export GGUF trên Colab (nếu đủ RAM/disk) hoặc **tải adapter/merged về máy** rồi convert local — chọn một nhánh ổn định theo tài liệu Unsloth/llama.cpp tại thời điểm bạn làm.
- Bước tiếp theo: mục **6–7** (GGUF → Ollama/LM Studio → VS Code).

### 2.9 Máy local / RunPod (tuỳ chọn, không bắt buộc nếu chỉ train Colab)

- Dùng khi cần session dài, GPU mạnh hơn, hoặc convert GGUF thoải mái hơn.
- CUDA + PyTorch khớp driver; giữ `requirements.txt` sau khi chạy thử thành công.

**Checkpoint (Colab):** `nvidia-smi` hiển thị GPU; `from_pretrained` load Qwen 4-bit xong; 1 batch train chạy không OOM; đã mount Drive và lưu thử được 1 checkpoint xuống Drive.

---

## 3. Chuẩn bị dataset (trọng tâm)

1. **Định dạng:** JSONL; mỗi dòng một object — trường khớp với pipeline Unsloth/TRL (instruction tuning hoặc chat messages).
2. **Nguồn:**
   - Open: dataset instruction code trên Hugging Face (đọc license).
   - Riêng: export snippet/repo (đã **lọc secret**, bỏ `.env`, token).
3. **Synthetic (tuỳ chọn):** script gọi API sinh cặp (prompt → code/giải thích) từ file trong repo; kiểm tra thủ công mẫu đầu ra.
4. **Chia tách:** train / validation (ví dụ 90/10); **không** lộ câu eval vào train.

**Checkpoint:** File `train.jsonl` + `val.jsonl`; 1 notebook/script load được và in được 3 mẫu đầu.

---

## 4. Fine-tune với QLoRA (Unsloth)

1. `FastLanguageModel.from_pretrained(model_name="Qwen/...", load_in_4bit=True, ...)`.
2. Gọi `get_peft_model` / cấu hình LoRA theo notebook mẫu (rank, target modules — thường notebook Qwen đã gợi ý).
3. `SFTTrainer` (hoặc API tương đương phiên bản TRL bạn dùng): trỏ dataset, `max_seq_length`, learning rate, epochs.
4. Lưu **adapter** (và log loss trên val).

**Checkpoint:** Loss val xuống hoặc ổn định; không overfit nặng (so với baseline trên vài prompt tay).

---

## 5. Đánh giá (trước khi merge / export)

1. **Bộ prompt cố định** (20–50 câu): sinh code, sửa lỗi, refactor, test — đúng stack bạn cần.
2. So sánh **base Qwen** vs **sau fine-tune** (cùng temperature, cùng template).
3. Ghi nhận: hallucination API, lỗi cú pháp, vi phạm convention.

**Checkpoint:** Bảng pass/fail hoặc điểm subjective có cấu trúc; quyết định epoch / data round tiếp theo.

---

## 6. Merge (tuỳ chọn) & xuất GGUF

1. **Adapter-only:** nhẹ, đổi base + adapter khi serve (phức tạp hơn với Ollama đơn giản).
2. **Merge weights:** một checkpoint đầy đủ — thuận tiện cho một số pipeline export.
3. **GGUF:** dùng công cụ / notebook được cộng đồng khuyến nghị cho đúng nhánh Qwen (llama.cpp, llama.cpp-python, hoặc script kèm Unsloth — cập nhật theo tài liệu hiện tại).

**Checkpoint:** File `.gguf` mở được trong LM Studio hoặc import Ollama thành công.

---

## 7. Inference local & VS Code

1. **Ollama** hoặc **LM Studio:** load GGUF; đặt context và template khớp Qwen (một số UI tự nhận; nếu sai, chỉnh manual).
2. **VS Code:** Continue.dev / Twinny — trỏ URL local (ví dụ `http://localhost:11434` với Ollama), chọn model vừa tạo.
3. Thử: giải thích đoạn code, sinh unit test, sửa bug có chủ đích.

**Checkpoint:** Luồng “bôi đen → prompt → nhận code” chạy ổn trên máy bạn.

---

## 8. Vận hành & phiên bản

| Thành phần | Versioning gợi ý |
|------------|------------------|
| Base HF ID | Ghi trong README / config |
| Dataset | `dataset_v1.jsonl`, `v2`, … |
| Adapter / GGUF | `my-qwen-code-v1.gguf` + ngày + hash (tuỳ chọn) |
| Prompt eval | `eval_prompts_v1.json` |

Khi có convention mới hoặc framework mới: **bổ sung mẫu** → train vòng ngắn → eval lại → tăng version.

---

## 9. Rủi ro thường gặp (Qwen + QLoRA)

| Hiện tượng | Hướng xử lý |
|------------|-------------|
| OOM | Giảm `max_seq_length`, batch, hoặc model nhỏ hơn; giữ 4-bit |
| Template / special tokens sai | Dùng tokenizer.apply_chat_template đúng bản Instruct |
| Thư viện cũ, không load Qwen3 | Upgrade transformers + Unsloth theo doc mới nhất |
| Phiên Colab **ngắt** giữa chừng | Lưu checkpoint định kỳ lên Drive; train từng đoạn; cân nhắc Colab Pro |

---

## 10. Thứ tự thực hiện gợi ý (checklist ngắn)

- [ ] Chốt **Model ID** Qwen (Coder/Instruct) + size theo VRAM Colab (`nvidia-smi`)  
- [ ] Colab: bật GPU, copy notebook Unsloth, cài pip, restart nếu cần, load model 4-bit thành công  
- [ ] Mount **Drive** + `output_dir` / `HF_HOME` (tuỳ chọn) để không mất checkpoint  
- [ ] Có `train.jsonl` / `val.jsonl` và đã rà secret  
- [ ] Chạy QLoRA (Unsloth) ít nhất 1 vòng ngắn (sanity)  
- [ ] Eval so với base  
- [ ] Export GGUF → Ollama/LM Studio  
- [ ] Nối VS Code (Continue/Twinny)  
- [ ] Ghi version dataset + model cho lần sau  

---

## 11. Tham chiếu nhanh (cập nhật khi làm)

- Hugging Face Qwen: https://huggingface.co/Qwen  
- Unsloth (notebook / doc fine-tune): tìm trên GitHub `unslothai/unsloth` và notebook Colab Qwen Coder  
- GGUF / llama.cpp: theo hướng dẫn phiên bản tương thích Qwen tại thời điểm export  

---

*Tài liệu này là lộ trình quy trình; siêu tham số cụ thể (LR, rank, epochs) nên lấy từ notebook Unsloth mới nhất cho đúng phiên bản thư viện và model bạn chọn.*
