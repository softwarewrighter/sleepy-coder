# References

## Core Papers

### Share: Shared LoRA Subspaces for Continual Learning
- **Title**: Shared LoRA Subspaces for almost Strict Continual Learning
- **arXiv**: [2602.06043v1](https://arxiv.org/abs/2602.06043)
- **HTML**: [https://arxiv.org/html/2602.06043v1](https://arxiv.org/html/2602.06043v1)

Key contributions:
- Continual LoRA with a single evolving shared low-rank basis
- Maintains tiny per-task coefficients instead of full adapters
- SVD-based merge/expand operations for adding new tasks
- Claims large parameter/memory savings vs classic "many adapters"
- Compatible with HuggingFace PEFT

### UWSH: Universal Weight Subspace Hypothesis
- **Title**: Universal Weight Subspace Hypothesis
- **arXiv**: [2512.05117](https://arxiv.org/abs/2512.05117)

Key contributions:
- Models of the same architecture converge to similar low-dimensional spectral subspaces
- Subspaces are stable across tasks, initializations, and modalities
- Enables reusing subspaces for adaptation and model merging
- Implies continual learning can be mostly coefficient updates in frozen basis

---

## Related Work

### O-LoRA: Orthogonal LoRA
- Keeps task subspaces orthogonal to reduce interference
- Strong forgetting guardrail

### KeepLoRA
- Projects gradients into residual/orthogonal subspaces
- Preserves prior knowledge while maintaining plasticity

### SPARC
- Continual prompt tuning in a low-dimensional PCA subspace
- Optionally combined with LoRA
- Good for lightweight demos

### C-LoRA
- Explores "can a single LoRA replace multiple LoRAs?"
- Mechanisms to control forgetting

### ShareLoRA (Different from Share)
- Another shared/robust direction
- Reports continual fine-tuning improvements on coding benchmarks (HumanEval)

---

## Background

### PaCT: Parameter-Efficient Continual Finetuning
- General term for continual finetuning using PEFT techniques
- Share explicitly frames itself as a PaCT approach
- Goal: "(almost strict) continual learning" without replay and without many adapters

### LoRA: Low-Rank Adaptation
- **Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Foundation for parameter-efficient fine-tuning
- Adds trainable low-rank matrices to frozen model weights

### PEFT: Parameter-Efficient Fine-Tuning
- **Repository**: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- HuggingFace library for LoRA, adapters, prompt tuning
- Share paper claims PEFT compatibility

---

## Tools and Libraries

### HuggingFace Transformers
- [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- Model loading and inference

### HuggingFace PEFT
- [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- LoRA training and merging

### Ollama
- [https://ollama.ai](https://ollama.ai)
- Local LLM inference

### vLLM
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- High-performance inference

---

## Inspiration

### Pi Minimal Agent
- Minimal tool-driven agent architecture
- Used in OpenClaw (formerly ClaudBot)
- Design principle: keep the core loop simple

### opencode
- Open-source coding agent
- Potential base for local LLM integration
