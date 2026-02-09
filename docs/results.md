# Sleepy-Coder Evaluation Results

*Generated: 2026-02-08 19:55:15*

## Summary

| Cycle | Model | Pass Rate | Passed | Failed | Median Steps |
|-------|-------|-----------|--------|--------|--------------|
| 0 | qwen2.5-coder:1.5b-instruct-q4_K_M | 76.7% | 23 | 7 | 2.0 |
| 1 | sleepy-coder-v2 | 60.0% | 18 | 12 | 2.0 |

## Analysis

**Regression: -16.7%** (76.7% → 60.0%)

### Possible Causes of Regression:

1. **Insufficient training data**: Only 23 failed episodes used for training
2. **Overfitting to training distribution**: Model may have overfit to specific error patterns
3. **Catastrophic forgetting**: Fine-tuning may have degraded general capabilities
4. **Training hyperparameters**: Learning rate or steps may need tuning
5. **Data quality**: Training on failed examples may include poor patterns

### What the Research Says

The regression we observed is a well-documented phenomenon called **catastrophic forgetting**. According to [2024 EMNLP research](https://aclanthology.org/2024.findings-emnlp.249/), there is a direct link between the flatness of the model's loss landscape and the extent of catastrophic forgetting.

Key findings from the literature:

1. **LoRA doesn't prevent forgetting**: While LoRA keeps the backbone frozen, [research shows](https://arxiv.org/abs/2403.01244) significant performance drops still occur when fine-tuning on sequential datasets.

2. **Replay outperforms regularization**: [Studies consistently show](https://openreview.net/pdf?id=IgZWU75BLL) that replay-based methods (mixing old + new data) outperform regularization-based methods like EWC and O-LoRA.

3. **RL generalizes better than SFT**: [2024 research](https://arxiv.org/html/2508.16546v1) indicates that RL methods (PPO) exhibit better generalization compared to supervised fine-tuning, which tends to memorize training data.

4. **Self-Synthesized Rehearsal (SSR)**: [ACL 2024](https://aclanthology.org/2024.acl-long.77/) proposes using the LLM itself to generate synthetic instances for rehearsal, achieving superior performance while being more data-efficient.

### Why Our Approach Failed

Our current implementation has several issues the literature warns about:

| Issue | Our Implementation | What Research Recommends |
|-------|-------------------|-------------------------|
| Training data | Only failed examples (23 episodes) | Mix of successful + failed examples |
| Data replay | None | Include base model training data |
| Method | Pure SFT | RL-based (PPO, DPO) or SSR |
| Data volume | 23 examples | Hundreds to thousands |
| Regularization | None | EWC, sharpness-aware minimization |

### Recommended Next Steps (Research-Backed)

1. **Add replay buffer**: Mix successful episodes with failed ones (recommended ratio 1:1 or higher)
2. **Use Self-Synthesized Rehearsal**: Generate synthetic training data from the base model
3. **Try RL-based fine-tuning**: Use DPO or PPO instead of pure SFT
4. **Apply EWCLoRA**: Combine EWC with LoRA to preserve important weights
5. **Sharpness-aware minimization**: Flatten loss landscape during training
6. **Generate more data**: Run many more evaluation cycles before training
7. **Lower learning rate**: Try 1e-4 or 5e-5 instead of 2e-4
8. **Quality filtering**: Only train on high-quality corrections, not all failures

## Results Plot

![Evaluation Results](results.png)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-Coder-1.5B-Instruct |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Training Steps | 500 |
| Learning Rate | 2e-4 |
| Batch Size | 4 |
| Quantization | 4-bit NF4 (QLoRA) |

## Raw Data

```json
{"cycle": 0, "error_signatures": {"max_attempts_exceeded": 7}, "failed": 7, "median_steps_to_green": 2.0, "model": "qwen2.5-coder:1.5b-instruct-q4_K_M", "pass_rate": 0.7666666666666667, "passed": 23, "run_id": "eval_cycle0_20260209_035102"}
{"cycle": 1, "error_signatures": {"max_attempts_exceeded": 12}, "failed": 12, "median_steps_to_green": 2.0, "model": "sleepy-coder-v2", "pass_rate": 0.6, "passed": 18, "run_id": "eval_cycle1_20260209_035157"}
```