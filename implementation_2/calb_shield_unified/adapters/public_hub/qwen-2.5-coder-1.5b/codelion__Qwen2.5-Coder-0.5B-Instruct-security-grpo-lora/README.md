---
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- ellora
- lora
- security
- secure-code
- vulnerability-prevention
- grpo
- preference-learning
- semgrep
- owasp
- cwe
- peft
- code-generation
- python
library_name: peft
license: apache-2.0
language:
- en
- code
pipeline_tag: text-generation
inference: true
model_type: qwen2
datasets:
- codelion/Qwen2.5-Coder-0.5B-Instruct-security-preference
---

# codelion/Qwen2.5-Coder-0.5B-Instruct-security-grpo-lora

## 🔐 Security-First Code Generation LoRA

This LoRA adapter enhances Qwen/Qwen2.5-Coder-0.5B-Instruct to generate secure code by default, trained using GRPO (Group Relative Policy Optimization) with automated security analysis via Semgrep.

## 🎯 Key Features

- **Automated Security Analysis**: Uses Semgrep for consistent vulnerability detection
- **Self-Supervised Training**: No manually curated secure/insecure datasets required
- **Comprehensive Coverage**: Addresses OWASP Top 10 and CWE Top 25 vulnerabilities
- **Language Focus**: Specialized for Python security patterns
- **Preference Learning**: GRPO training to prefer secure coding patterns

## 📊 Performance Metrics

- **Base Model**: Qwen/Qwen2.5-Coder-0.5B-Instruct
- **Training Method**: GRPO with security-based preferences
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Training Samples**: 195
- **Security Evaluation Pass Rate**: 20.0%
- **Average Security Score**: 0.40 (lower is better)

### Vulnerability Prevention Results

| Vulnerability Type | Score | Status |
|-------------------|-------|---------|
| SQL Injection | 0 | ✅ |
| Command Injection | 0 | ✅ |
| Path Traversal | 2 | ✅ |
| Weak Cryptography | 0 | ✅ |
| Hardcoded Secrets | 0 | ✅ |

## 🔧 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Load security LoRA adapter
model = PeftModel.from_pretrained(model, "codelion/Qwen2.5-Coder-0.5B-Instruct-security-grpo-lora")

# Generate secure code
prompt = '''Write a secure Python function: Create a user login function
that checks username and password against a database'''

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
secure_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(secure_code)
```

## 📈 Expected Output

The model generates code with security best practices:

```python
def login_user(username, password):
    """Securely authenticate a user against the database."""
    import bcrypt
    import secrets
    from sqlalchemy import text

    # Validate inputs
    if not username or not password:
        return False, "Invalid credentials"

    # Use parameterized query to prevent SQL injection
    query = text("SELECT user_id, password_hash FROM users WHERE username = :username")
    result = db.execute(query, {"username": username}).fetchone()

    if not result:
        # Prevent timing attacks by still checking a dummy password
        bcrypt.checkpw(b"dummy", b"$2b$12$dummy.hash.to.prevent.timing")
        return False, "Invalid credentials"

    # Verify password using bcrypt
    if bcrypt.checkpw(password.encode('utf-8'), result.password_hash):
        # Generate secure session token
        session_token = secrets.token_urlsafe(32)
        return True, session_token

    return False, "Invalid credentials"
```

## 🛡️ Security Patterns Learned

- **SQL Injection Prevention**: Parameterized queries, prepared statements
- **Password Security**: Bcrypt/Argon2 hashing, no plaintext storage
- **Input Validation**: Comprehensive validation and sanitization
- **Error Handling**: Safe error messages without information disclosure
- **Secure Randomness**: Using `secrets` module instead of `random`
- **Path Security**: Proper path joining and validation
- **Command Injection Prevention**: Avoiding shell=True, using subprocess safely

## 🧪 Training Details

### Data Generation
- **Method**: Self-supervised with Magpie-style generation
- **Scenarios**: 7 security categories
- **Analysis**: Automated using Semgrep security rules
- **Preference Pairs**: Based on security score differences

### GRPO Training
- **Objective**: Minimize security vulnerabilities while maintaining functionality
- **Reward Signal**: Negative correlation with Semgrep security score
- **Batch Size**: 1 with 8x gradient accumulation
- **Learning Rate**: 3e-06
- **Epochs**: 5

## 📚 Evaluation

The adapter was evaluated on comprehensive security test cases:

- **CWE Coverage**: Top 25 most dangerous software weaknesses
- **OWASP Alignment**: Addresses OWASP Top 10 vulnerabilities
- **Practical Scenarios**: Real-world security challenges
- **Pattern Recognition**: Identifies and applies secure coding patterns

## 🔍 Limitations and Considerations

1. **Language Focus**: Currently optimized for Python; other languages may need additional training
2. **Context Awareness**: Best results with clear security-focused prompts
3. **Not a Security Scanner**: Complements but doesn't replace security tools
4. **Continuous Updates**: Security landscape evolves; periodic retraining recommended

## 🔗 Related Resources

- **Dataset**: [codelion/Qwen2.5-Coder-0.5B-Instruct-security-preference](https://huggingface.co/datasets/codelion/Qwen2.5-Coder-0.5B-Instruct-security-preference)
- **Base Model**: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- **Ellora Project**: [GitHub Repository](https://github.com/codelion/ellora)
- **Semgrep**: [Security Analysis Tool](https://semgrep.dev/)

---

*This adapter is part of the [Ellora project](https://github.com/codelion/ellora) - standardized recipes for enhancing LLM capabilities.*
