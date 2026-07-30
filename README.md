# AI Engineering Knowledge Wiki

[![tests](https://github.com/navinsreyas/knowledge-wiki/actions/workflows/tests.yml/badge.svg)](https://github.com/navinsreyas/knowledge-wiki/actions/workflows/tests.yml)

A local, end-to-end pipeline that turns a self-directed AI Engineering study plan into a structured knowledge wiki, fine-tunes a model on it, and evaluates whether fine-tuning actually helps.

Everything runs on consumer hardware (RTX 4060 Laptop GPU, 8GB VRAM). The project ingests PDFs, papers, YouTube transcripts, and web articles, compiles them into linked wiki articles inside an Obsidian vault, generates a synthetic question-answer dataset from that wiki, QLoRA fine-tunes Qwen2.5:7B, and runs a three-way benchmark comparing the base model, the fine-tuned model, and retrieval-augmented generation (RAG) over the wiki.

## What it does

The pipeline has two halves.

**Wiki construction.** Source material is ingested and deduplicated, then compiled by a local LLM into structured articles organised by topic. A graph builder links related concepts across articles. The result is a browsable Obsidian vault of 163 articles spanning maths, deep learning, fine-tuning, RAG, agents, evaluation, serving, MLOps, and more.

**Fine-tuning and evaluation.** A synthetic QA dataset is generated from the compiled articles with an LLM grounding-validation step, then split into train/validation/test. Qwen2.5:7B is QLoRA fine-tuned on the training split, quantized to GGUF, and served locally through Ollama. A held-out test set drives a three-way comparison of base, fine-tuned, and RAG.

## Results

Verified on 42 held-out QA pairs:

| Method | Accuracy |
| --- | --- |
| Base Qwen2.5:7B | 76.2% |
| Fine-tuned (QLoRA) | 45.2% |
| RAG over wiki | 33.3% |

Fine-tuning on 326 synthetic pairs reduced accuracy from 76.2% to 45.2%, consistent with catastrophic forgetting at this data scale. Single run, n=42, with the judge drawn from the same model family as the candidate — treat as directional, not conclusive.

The RAG score reflects a retrieval-relevance gap, not a broken pipeline — retrieval fires and returns articles, but keyword matching pulls weakly-related articles for many queries. Upgrading to embedding-based retrieval is the obvious next step.

These numbers come from a single run with n=42 and a judge from the same model family as two of the three systems under test. They are directional, not definitive.

## Honest notes

- The Ollama-served fine-tuned model exhibits generation instability (token repetition) attributable to the training configuration. The 45.2% figure was measured through a direct inference path with the correct chat template, not through the unstable Ollama deployment.
- An earlier evaluation run produced different numbers that were later found invalid: the fine-tuned model had been fed the wrong prompt format, and the RAG path was not retrieving at all after a folder restructure. Both were fixed before the numbers above were finalised.
- The synthetic QA generator validates pairs against only the first 1,500 characters of each source article, uses the same model to generate and validate (self-validation bias), and does not persist rejection counts. The method is reproducible; a precise pass rate is not claimable.

## Stack

Python, Qwen2.5:7B via Ollama, Unsloth (QLoRA training), llama.cpp (GGUF quantization), Obsidian Local REST API (vault I/O), NetworkX (concept graph), LangChain (RAG orchestration).

## Repository layout

```
pipeline/     ingestion, compilation, query, linting, graph building, CLI
finetune/     QA generation, training, evaluation, data splits
config.py     configuration
requirements.txt
```

The Obsidian vault (personal study content), model weights, and build artifacts are kept local and are not tracked in this repository.

## Running it

Requires a local Ollama install with `qwen2.5:7b` pulled, an Obsidian vault with the Local REST API plugin enabled, and the Python dependencies in `requirements.txt`. Pipeline commands run through the `wiki` CLI (ingest, compile, query, lint). Fine-tuning and evaluation run through the scripts in `finetune/`.

This is a personal learning project. It is shared as a demonstration of an end-to-end fine-tuning and evaluation workflow, including the parts where the fancy method did not win.
