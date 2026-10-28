---
language:
- vi
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-1.5B-Instruct
- lora
- sft
- transformers
- trl
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
pipeline_tag: text-generation
library_name: peft
---

# Qwen2.5-Coder 1.5B Instruct - Fine-tuned Vietnamese Ethnic Text-to-SQL

Đây là mô hình LoRA Adapter được huấn luyện dựa trên nền tảng **Qwen/Qwen2.5-Coder-1.5B-Instruct** nhằm thực hiện nhiệm vụ dịch văn bản câu hỏi tiếng Việt sang câu lệnh SQL (SQLite) chuyên sâu về cơ sở dữ liệu **Danh sách các Dân tộc Việt Nam**.

## 📊 Thông tin tập dữ liệu huấn luyện
- **Nguồn dữ liệu**: Danh sách dân tộc Việt Nam (bao gồm các trường dữ liệu: `STT`, `TenDanToc`, `DanSo2019`, `TyLe`).
- **Kích thước**: Tạo tự động 111 cặp câu hỏi - đáp câu lệnh SQL dạng ChatML.

## 🛠️ Phương pháp huấn luyện
- **Kiến trúc**: QLoRA (Quantized Low-Rank Adaptation) 4-bit với cấu hình:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.05`
- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Optimizer**: `paged_adamw_8bit` ổn định hóa hiệu năng trên T4 GPU.

## 🚀 Hướng dẫn sử dụng nhanh

Bạn có thể dễ dàng tải và sử dụng mô hình này bằng thư viện `transformers` và `peft`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_id = "vinhdinh-star/qwen2.5-coder-1.5b-dantoc-vietnam"

# 1. Cấu hình lượng tử hóa 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 2. Tải Tokenizer và Base Model
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# 3. Nạp LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

def ask_text_to_sql(question):
    prompt = f"<|im_start|>system
Bạn là chuyên gia Text-to-SQL. Dựa vào bảng DanhSachDanToc (STT, TenDanToc, DanSo2019, TyLe), hãy viết câu lệnh SQLite.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
SELECT "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = "SELECT " + raw_output.split("SELECT")[-1].strip()
    return sql_query

# Chạy thử
print(ask_text_to_sql("Top 5 dân tộc đông dân nhất Việt Nam"))
```

## 📝 Liên hệ và Đóng góp
Mô hình được phát triển bởi **VinhDinh Star**. Mọi góp ý hoặc phản hồi vui lòng gửi qua mục Discussion trên repository này!
### Framework versions

- PEFT 0.21.2