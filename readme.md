<div align="center">

<img src="STaR.jpg" alt="STaR" width="420"/>

# STaR: Slow-Thinking for Table Reasoning (Anonymous)

</div>

## Abstract

Table reasoning with the large language models (LLMs) is a fundamental path toward building intelligent systems that can understand and analyze over structured data. While recent progress has shown promising results, they still suffer from two key limitations: (i) the reasoning processes lack the depth and iterative refinement characteristic of human cognition; and (ii) the reasoning processes exhibit instability, which compromises their reliability in downstream applications. In this work, we present STaR (slow-thinking for table reasoning), a new framework achieving cognitive table reasoning, in which LLMs are equipped with slow-thinking capabilities by explicitly modeling step-by-step thinking and uncertainty-aware inference. During training, STaR employs two-stage difficulty-aware reinforcement learning (DRL), progressively learning from simple to complex queries under a composite reward. During inference, STaR performs trajectory-level uncertainty quantification by integrating token-level confidence and answer consistency, enabling selection of more credible reasoning paths. Extensive experiments on benchmarks demonstrate that STaR achieves superior performance and enhanced reasoning stability. Moreover, strong generalization over out-of-domain datasets further demonstrates STaR's potential as a reliable and cognitively inspired solution for table reasoning with LLMs.

This repository provides the anonymous implementation of STaR based on the `verl` framework.

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
