<div align="center">

<img src="STaR.jpg" alt="STaR" width="420"/>

# STaR: Slow-Thinking for Table Reasoning (Anonymous)

</div>

## Abstract

Table reasoning with the large language models (LLMs) is a fundamental path toward building intelligent systems that can understand and analyze over structured data. While recent progress has shown promising results, they still suffer from two key limitations: (i) the reasoning processes lack the depth and iterative refinement characteristic of human cognition; and (ii) the reasoning processes exhibit instability, which compromises their reliability in downstream applications. In this work, we present STaR (slow-thinking for table reasoning), a new framework achieving cognitive table reasoning, in which LLMs are equipped with slow-thinking capabilities by explicitly modeling step-by-step thinking and uncertainty-aware inference. During training, STaR employs two-stage difficulty-aware reinforcement learning (DRL), progressively learning from simple to complex queries under a composite reward. During inference, STaR performs trajectory-level uncertainty quantification by integrating token-level confidence and answer consistency, enabling selection of more credible reasoning paths. Extensive experiments on benchmarks demonstrate that STaR achieves superior performance and enhanced reasoning stability. Moreover, strong generalization over out-of-domain datasets further demonstrates STaR's potential as a reliable and cognitively inspired solution for table reasoning with LLMs.

## Installation

Tested with Python 3.10 and CUDA GPUs.

```bash
# 1) Clone (anonymous placeholder URL)
git clone [REPO_URL]
cd ano-repo

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Install verl in editable mode
cd verl
pip install -e .
cd -
```

## Training

Shell scripts are under `sh/`. Adjust paths and hyperparameters inside the scripts as needed.

- SFT
  - `bash sh/STaR-sft-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-qwen3-8b.sh`

- GRPO — Stage 1
  - `bash sh/STaR-sft-stage1-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-stage1-qwen3-8b.sh`

- GRPO — Stage 2
  - `bash sh/STaR-sft-stage1-stage2-qwen3-0.6b.sh`
  - `bash sh/STaR-sft-stage1-stage2-qwen3-8b.sh`

## Evaluation

1) Run rollout to generate trajectories:

```bash
bash sh/STaR-eval.sh
```

2) Compute EM metric by trajectory:

```bash
python eval-by-trajectory.py
```

## Acknowledgements

This work builds on the excellent `verl` framework. We thank the community for open-source tools and datasets used in our experiments.

## Prompt Templates

Below we list the exact prompt templates used in training.

### SFT Prompt (Single Message)

The SFT data uses a single text prompt with the following structure (placeholders in square brackets are filled with the actual example):

````text
Instruction
Answer the question based on the provided table.


Table
Table Title: [TABLE_TITLE]
Table Content:
[TABLE_CONTENT]


Question
[QUESTION]


Answer Format
The final answer should be concise and use the following format:
```json
{
  "answer": [
    "answer1",
    "answer2",
    ...
  ]
}
```
````

### GRPO Prompt (Two-Stage Training)

Both GRPO stages (Stage 1 and Stage 2) use the same Chat-style prompt; the two stages only differ in the reward design. The model input is a list of messages:

````text
[
  {
    "role": "system",
    "content": "A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
  },
  {
    "role": "user",
    "content": "Instruction\nAnswer the question based on the provided table.\n\n\nTable\nTable Title: [TABLE_TITLE]\nTable Content:\n[TABLE_CONTENT]\n\n\nQuestion\n[QUESTION]\n\n\nAnswer Format\nThe final answer should be concise and use the following format:\n```json\n{\n  \"answer\": [\n    \"answer1\",\n    \"answer2\",\n    ...\n  ]\n}\n```"
  }
]
````

### Table Serialization Format

In all experiments, the placeholder `[TABLE_CONTENT]` is instantiated as a **GitHub‑flavored Markdown table string** (plain-text Markdown, not JSON/CSV/HTML):

- The table is rendered as Markdown using the pipe (`|`) syntax with one header row, one alignment row (`| --- | ... |`), and one row per record, exactly as in the example in the paper.
- Column order in `[TABLE_CONTENT]` matches the original dataset schema; we do not reorder columns.
- Row order is preserved from the underlying dataset without sorting or filtering (unless explicitly stated for a benchmark).
- Cell values are inserted as-is from the dataset (including commas, percentage signs, and other punctuation); missing cells are left empty.
- The line `Table Title: [TABLE_TITLE]` is plain text, where `[TABLE_TITLE]` is the human-readable table name from the dataset.

### Decoding and Sampling Parameters

We summarize the decoding configuration used in our main results; all values are also encoded in the provided shell scripts.

- Evaluation (script `sh/STaR-eval.sh`):
  - Number of stochastic passes per example: `N_PASSES = 8` (`data.n_samples=8`).
  - Temperature: `0.6` (`rollout.temperature=0.6`).
  - Top-p: `0.95` (`rollout.top_p=0.95`).
  - Top-k: disabled (`rollout.top_k=-1`).
  - Max prompt length: `8192` tokens (`rollout.prompt_length=8192`).
  - Max response length: `4096` tokens (`rollout.response_length=4096`).
  - Batch size during generation: `1024` examples (`data.batch_size=1024`).
  - For each example, all samples, log-probabilities, and entropies are saved; the final answer is selected by `eval-by-trajectory.py` using the consistency–confidence fusion rule.

- GRPO training (scripts `sh/STaR-sft-stage1-*.sh`, `sh/STaR-sft-stage1-stage2-*.sh`):
  - Number of sampled trajectories per prompt: `rollout_n = 5` (Stage 1) or `rollout_n = 8` (Stage 2).
  - Training sampling temperature: `train_temperature = 1.0`.
  - Validation sampling temperature: `val_temperature = 0.6` with `do_sample=True`, `n=1`.
  - Other decoding hyperparameters not explicitly set in these scripts use the defaults of the `verl` framework.
