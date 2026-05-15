# Qwen fine-tune — lộ trình & quyết định (Colab)

Repo tài liệu + quy trình huấn luyện model code (base **Qwen / Qwen-Coder**) trên **Google Colab** (Unsloth + QLoRA), sau đó xuất GGUF và chạy local nếu cần.

## Nội dung

| File | Mô tả |
|------|--------|
| [docs/LO_TRINH_QWEN_BASE.md](docs/LO_TRINH_QWEN_BASE.md) | Lộ trình đầy đủ: HF, dataset, Colab, eval, GGUF, VS Code |
| [docs/QUYET_DINH_VA_THUC_HIEN.md](docs/QUYET_DINH_VA_THUC_HIEN.md) | Quyết định đã chốt + lý do + Model ID mặc định Colab T4 |
| [docs/COLAB_SAU_KHI_CLONE.md](docs/COLAB_SAU_KHI_CLONE.md) | **Bước tiếp theo sau `git clone` trên Colab** (GPU, Drive, HF, Unsloth, train thử) |
| [notebooks/Hole_Qwen25_Coder_1.5B_Alpaca.ipynb](notebooks/Hole_Qwen25_Coder_1.5B_Alpaca.ipynb) | Notebook Colab: cài Unsloth → restart → Drive → train 60 bước → lưu LoRA |

## Remote GitHub

- Repo: [github.com/pendragonIV/Hole](https://github.com/pendragonIV/Hole)
- Remote `origin` trên máy bạn: `https://github.com/pendragonIV/Hole.git`

## Dùng trên Google Colab

Sau khi đã **push** code lên GitHub, trong Colab clone repo rồi đọc **[docs/COLAB_SAU_KHI_CLONE.md](docs/COLAB_SAU_KHI_CLONE.md)** để làm các bước tiếp theo (GPU, Drive, Unsloth, train thử).

```python
!git clone https://github.com/pendragonIV/Hole.git
%cd Hole
# !ls docs/
```

Hoặc **Upload** / **Mount Google Drive** rồi mở file `.md` trong repo để đọc song song với notebook **Unsloth** (khuyến nghị: lấy notebook gốc từ [unslothai/unsloth](https://github.com/unslothai/unsloth) rồi chỉnh `model_name` theo `docs/QUYET_DINH_VA_THUC_HIEN.md`).

### Notebook train smoke (Open in Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pendragonIV/Hole/blob/main/notebooks/Hole_Qwen25_Coder_1.5B_Alpaca.ipynb)

Hoặc clone repo rồi mở `notebooks/Hole_Qwen25_Coder_1.5B_Alpaca.ipynb`. Chi tiết từng bước: [docs/COLAB_SAU_KHI_CLONE.md](docs/COLAB_SAU_KHI_CLONE.md).

## Không đưa lên Git

Xem [.gitignore](.gitignore): token, `.env`, weight lớn (`.gguf`, `*.safetensors`, …) — tránh lộ secret và vượt giới hạn GitHub.

## License

Nội dung tài liệu trong repo: bạn tự đặt license khi tạo repo trên GitHub (ví dụ MIT nếu chỉ là markdown hướng dẫn).
