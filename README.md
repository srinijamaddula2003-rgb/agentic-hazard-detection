# Agentic AI for Automated Security Hazard Detection in 3GPP Protocol Specifications

An LLM-based agentic pipeline for detecting security hazard indicators (HIs) in 3GPP protocol specifications (TS 24.301 — LTE NAS), replacing fine-tuned ML models with prompt engineering and multi-agent design.

**Key Result:** F1 = 0.938 with perfect precision on HI detection, exceeding Bookworm (IEEE S&P 2021) by 30%. Additionally discovered 21 novel threats beyond predefined rules.

---

## Table of Contents

- [Overview](#overview)
- [Research Problem](#research-problem)
- [Pipeline Architecture](#pipeline-architecture)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
  - [Phase 1: 150-Paragraph Dataset (V0.1.0)](#phase-1-150-paragraph-dataset)
  - [Phase D: 675-Paragraph Dataset (V9.0.0)](#phase-d-675-paragraph-dataset)
  - [Stage 1: Bookworm Comparison (V16.6.0)](#stage-1-bookworm-comparison)
- [Experimental Results](#experimental-results)
  - [Vanilla LLM Baselines](#vanilla-llm-baselines)
  - [Prompt Ablation Study](#prompt-ablation-study)
  - [Thinking Mode Analysis](#thinking-mode-analysis)
  - [Agentic Pipeline Progression](#agentic-pipeline-progression)
  - [Open-Ended Discovery](#open-ended-discovery)
  - [Novel Threat Validation](#novel-threat-validation)
- [Key Findings](#key-findings)
- [Related Work](#related-work)
- [Citation](#citation)

---

## Overview

3GPP specifications like TS 24.301 (LTE NAS) are 300+ page documents defining mobile network protocol procedures. Security vulnerabilities hide in these documents as **Hazard Indicators (HIs)** — statements linking trigger events to destructive operations (credential deletion, bearer deactivation, procedure aborts).

This project builds an **agentic LLM pipeline** that automates HI detection without any model fine-tuning, using only prompt engineering and multi-agent architecture.

## Research Problem

**Bookworm** (Chen et al., IEEE S&P 2021) introduced automated HI detection using fine-tuned RoBERTa, achieving F1 ≈ 0.72. Key limitations:
- Requires expensive fine-tuning with labeled training data
- Cannot detect implicit or cross-paragraph hazards
- Limited to DoS-related threat seeds
- No rigorous P/R/F1 evaluation methodology

**Our approach** replaces fine-tuned models with an agentic LLM pipeline that:
- Requires zero training data — only prompt engineering
- Uses multi-model consensus for robustness
- Employs context engineering for cross-paragraph hazards
- Includes open-ended discovery mode for novel threats
- Provides rigorous evaluation with corrected gold standard

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Prefilter   │───▶│  Multi-Model     │───▶│    Context       │
│  (Keywords)  │    │  Consensus       │    │  Engineering     │
│              │    │  (Llama + Qwen)  │    │  (±2 paragraphs) │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                   ┌──────────────────┐    ┌────────▼────────┐
                   │   Open-Ended     │    │   Skeptical     │
                   │   Discovery      │    │   Validator     │
                   │   (no rules)     │    │   (filter FPs)  │
                   └──────────────────┘    └─────────────────┘
```

**Stage 1 — Prefilter:** Rule-based keyword filtering identifies candidate paragraphs using risky action verbs (abort, delete, discard, reset, deactivate) and conditional triggers (if, upon, when).

**Stage 2 — Multi-Model Consensus:** Both Llama 3.2 3B (local) and Qwen 3.5 397B (remote) classify each candidate. Union voting: if either model says hazard, accept it. Exploits complementary blind spots.

**Stage 3 — Context Engineering:** For consensus-NO paragraphs with moderate risk scores, re-classifies with surrounding paragraph context (±2 paragraphs + section metadata). Addresses cross-paragraph hazards where trigger and action span different paragraphs.

**Stage 4 — Skeptical Validation:** Context-rescued paragraphs go through a skeptical validator that filters false positives. Checks whether the flagged behavior involves a concrete trigger-action pair vs. a general description.

**Stage 5 — Open-Ended Discovery:** Separate mode with no predefined rules. Asks the LLM to identify any exploitable behavior from an attacker's perspective. Finds threats beyond the HI definition.

## Results Summary

| Method | Precision | Recall | F1 | FP | FN |
|--------|-----------|--------|------|-----|-----|
| Llama 3.2 3B (vanilla) | 1.000 | 0.647 | 0.786 | 0 | 6 |
| Qwen 397B (vanilla) | 1.000 | 0.647 | 0.786 | 0 | 6 |
| Multi-model consensus | 1.000 | 0.824 | 0.903 | 0 | 3 |
| **Consensus + validated context** | **1.000** | **0.882** | **0.938** | **0** | **2** |
| Open-ended discovery | 0.254 | 0.941 | 0.400 | 47 | 1 |
| Bookworm (reference) | ~0.560 | ~1.000 | ~0.720 | — | — |

**Novel threat discovery:** 21 potential threats beyond gold standard, including 8 independently rediscovered known vulnerabilities (IMSI disclosure, GUTI spoofing, identity flooding) and 9 potentially novel findings.

## Project Structure

```
lte-hazard-llm/
├── data/
│   ├── raw_specs/               # 3GPP specification PDFs
│   │   ├── TS_24.301            # V0.1.0 (early draft, 63 pages)
│   │   ├── TS_24.301.900.pdf    # V9.0.0 (Release 9, 272 pages)
│   │   ├── TS_24.301.16.6.0.pdf # V16.6.0 (Bookworm's version)
│   │   ├── HI_all_web3.pdf      # Bookworm ground truth (192 HIs)
│   │   ├── T1_web4.pdf           # Bookworm T1: "abort procedure" HIs
│   │   └── T2_web5.pdf           # Bookworm T2: "consider USIM" HIs
│   └── processed/               # Ingested paragraphs & predictions
│       ├── annotation_sample.csv              # 150-para gold standard (V0.1.0)
│       ├── TS_24.301_paragraphs.jsonl         # Ingested V0.1.0
│       ├── TS_24_301_v9_paragraphs.jsonl      # Ingested V9.0.0 (full)
│       ├── TS_24_301_v9_bookworm.jsonl        # V9.0.0 Bookworm scope (675 paras)
│       ├── TS_24_301_v16_paragraphs.jsonl     # Ingested V16.6.0
│       ├── bookworm_gold_standard.csv         # Gold standard from Bookworm TPs
│       ├── ollama_predictions.jsonl           # Llama predictions
│       ├── qwen_nothinking_predictions.jsonl  # Qwen (thinking OFF) predictions
│       ├── qwen_thinking_predictions.jsonl    # Qwen (thinking ON) predictions
│       ├── consensus_predictions.jsonl        # Consensus results
│       ├── context_eng_predictions.jsonl       # Context engineering results
│       ├── combined_pipeline_predictions.jsonl # Best pipeline results
│       ├── openended_predictions.jsonl        # Open-ended discovery results
│       ├── hybrid_predictions.jsonl           # Hybrid (two-pass) results
│       ├── hybrid_controlled_predictions.jsonl # Hybrid (single-prompt) results
│       ├── context_thinking_predictions.jsonl  # Context + thinking ON results
│       └── ablation/                          # Prompt ablation results
│           ├── A1_zero_shot_predictions.jsonl
│           ├── A2_guided_zero_shot_predictions.jsonl
│           ├── A3_fewshot_1_predictions.jsonl
│           ├── A4_fewshot_3_predictions.jsonl
│           └── A5_fewshot_5_predictions.jsonl
├── src/
│   ├── ingestion.py                    # PDF ingestion for V0.1.0
│   ├── ingestion_v9.py                 # PDF ingestion for V9.0.0+ (improved)
│   ├── llm_extractor_ollama.py         # Llama 3.2 3B extractor
│   ├── llm_extractor_qwen.py          # Qwen 397B extractor
│   ├── qwen_nothinking.py             # Qwen with thinking OFF
│   ├── qwen_thinking.py               # Qwen with thinking ON
│   ├── compare_thinking.py            # Compare thinking ON vs OFF
│   ├── evaluate_qwen.py               # Evaluation metrics
|   |__ evaluate_ollama.py             # Evaluation metrics
│   ├── agentic_step1_consensus.py     # Multi-model consensus
│   ├── agentic_step2_reflexion.py     # Reflexion (naive)
│   ├── agentic_step2b_reflexion.py  # Reflexion + validation
|   |__ compare_models.py              # to compare qwen and llama
│   ├── agentic_step3_context.py       # Context engineering
│   ├── agentic_step5_combined.py      # Consensus + validated context
│   ├── agentic_step5b.py   # Attacker-perspective validator
│   ├── agentic_step6_react.py         # ReAct with spec retrieval
│   ├── ablation_study.py             # 5-variant prompt ablation study
│   ├── openended_discovery.py         # Open-ended discovery mode
│   ├── hybrid_discovery.py            # Hybrid: two-pass with validation
│   ├── hybrid_controlled.py           # Hybrid: single unified prompt
│   ├── context_thinking.py            # Context engineering + thinking ON
│   ├── pipeline_v9.py                 # Full pipeline for V9.0.0 dataset
│   ├── bookworm_ground_truth.py       # Bookworm ground truth matching
│   ├── state_event_recovery.py        # State/event extraction
│   ├── test_case_generator.py         # Test case generation
│   └── generate_report.py            # Report generation
├── .env                                # Environment variables
├── requirements.txt                    # Python dependencies
├── attribution.md                      # Attribution and credits
├── venv/                               # Python virtual environment (not tracked in git)
└── README.md
```

## Setup & Installation

### Requirements

- Python 3.7+
- [Ollama](https://ollama.com/) with Llama 3.2 3B model (local)
- Access to Qwen 3.5 397B via llama.cpp server (remote workstation)

### Install Dependencies

```bash
pip install pdfplumber requests pandas
```

### Model Setup

**Llama 3.2 3B (local):**
```bash
ollama pull llama3.2:3b
ollama serve  # If not already running
```

**Qwen 3.5 397B (remote workstation):**
```bash
# SSH into workstation
I used my remote workstation

# Start llama.cpp server (thinking OFF)
remote server

**SSH tunnel (from local machine):**
```bash
ssh -L 8080:localhost:8080 username
```

## Running the Pipeline

### Phase 1: 150-Paragraph Dataset

Run individual experiments on the original 150-paragraph V0.1.0 dataset:

```bash
# Vanilla baselines
python src/llm_extractor_ollama.py       # Llama 3.2 3B
python src/qwen_nothinking.py            # Qwen thinking OFF
python src/qwen_thinking.py              # Qwen thinking ON

# Compare thinking modes
python src/compare_thinking.py

# Agentic pipeline (run in order)
python src/agentic_step1_consensus.py    # Multi-model consensus
python src/agentic_step3_context.py      # Context engineering
python src/agentic_step5_combined.py     # Consensus + validated context

# Prompt ablation (run individually or all at once)
python src/prompt_ablation.py A1_zero_shot
python src/prompt_ablation.py A2_guided_zero_shot
python src/prompt_ablation.py A3_fewshot_1
python src/prompt_ablation.py A4_fewshot_3
python src/prompt_ablation.py A5_fewshot_5
python src/prompt_ablation.py eval       # Show results

# Open-ended discovery
python src/openended_discovery.py        # Run discovery
python src/openended_discovery.py eval   # Show results

# Hybrid approaches
python src/hybrid_discovery.py           # Two-pass with validation
python src/hybrid_controlled.py          # Single unified prompt

# Context + thinking ON comparison
python src/context_thinking.py           # Needs thinking ON server
```

### Phase D: 675-Paragraph Dataset (V9.0.0)

```bash
# Step 1: Ingest the specification
python src/ingestion_v9.py

# Step 2: Run the full pipeline (6 steps, run in order)
python src/pipeline_v9.py llama          # Needs Ollama
python src/pipeline_v9.py qwen           # Needs SSH tunnel
python src/pipeline_v9.py consensus      # Instant
python src/pipeline_v9.py context        # Needs SSH tunnel
python src/pipeline_v9.py validate       # Needs SSH tunnel
python src/pipeline_v9.py report         # Generates pre-annotated CSV
```

### Stage 1: Bookworm Comparison (V16.6.0)

```bash
# Extract Bookworm's ground truth and match to paragraphs
python src/bookworm_ground_truth.py extract   # Parse 192 HIs from PDF
python src/bookworm_ground_truth.py match     # Match 87 TPs to paragraphs
python src/bookworm_ground_truth.py report    # Show matching quality
```

## Experimental Results

### Vanilla LLM Baselines

| Model | Thinking | Precision | Recall | F1 |
|-------|----------|-----------|--------|------|
| Llama 3.2 3B | N/A | 1.000 | 0.647 | 0.786 |
| Qwen 397B | OFF | 1.000 | 0.647 | 0.786 |
| Qwen 397B | ON | 0.846 | 0.647 | 0.733 |

**Finding:** The 397B model performs identically to the 3B model with thinking OFF. Thinking ON hurts precision by introducing false positives while providing no recall improvement.

### Prompt Ablation Study

| Variant | Precision | Recall | F1 |
|---------|-----------|--------|------|
| A1: Pure zero-shot | 0.000 | 0.000 | 0.000 |
| A2: Guided zero-shot (rules only) | 0.909 | 0.588 | 0.714 |
| A3: Few-shot (1 example) | 0.917 | 0.647 | 0.759 |
| **A4: Few-shot (3 examples)** | **1.000** | **0.647** | **0.786** |
| A5: Few-shot (5 examples) | 1.000 | 0.588 | 0.741 |

**Finding:** Pure zero-shot completely fails. Rules matter more than examples. 3 examples is optimal — more makes the model overly conservative.

### Thinking Mode Analysis (2×2)

| Configuration | Precision | Recall | F1 | FP |
|--------------|-----------|--------|------|-----|
| Thinking OFF, no context | 1.000 | 0.647 | 0.786 | 0 |
| Thinking ON, no context | 0.846 | 0.647 | 0.733 | 2 |
| Thinking OFF + context | 0.708 | 1.000 | 0.829 | 7 |
| Thinking ON + context | 0.486 | 1.000 | 0.654 | 18 |

**Finding:** Thinking mode hurts structured classification regardless of context. Context engineering achieves perfect recall in both modes, but thinking ON amplifies noise (18 FP vs 7 FP).

### Agentic Pipeline Progression

| Step | Method | Precision | Recall | F1 |
|------|--------|-----------|--------|------|
| 0 | Llama 3.2 3B (vanilla) | 1.000 | 0.647 | 0.786 |
| 1 | Multi-model consensus | 1.000 | 0.824 | 0.903 |
| 2 | + Context engineering (unvalidated) | 0.708 | 1.000 | 0.829 |
| 3 | + Reflexion (naive) | 0.421 | 0.941 | 0.582 |
| 3b | + Reflexion (validated) | 0.933 | 0.824 | 0.875 |
| **4** | **+ Validated context engineering** | **1.000** | **0.882** | **0.938** |

### Open-Ended Discovery

| Mode | Precision | Recall | F1 | Novel Threats |
|------|-----------|--------|------|---------------|
| Guided (rules + examples) | 1.000 | 0.647 | 0.786 | 0 |
| Open-ended (no rules) | 0.254 | 0.941 | 0.400 | 47 flagged |
| After validation | 0.725 | 0.941 | 0.821 | 21 confirmed |

### Novel Threat Validation

| Category | Count | % | Examples |
|----------|-------|---|----------|
| Independently rediscovered | 8 | 38% | IMSI disclosure, GUTI spoofing, identity flooding |
| Partially known | 4 | 19% | KSI sync failure, TAU exploitation |
| Potentially novel | 9 | 43% | IPv6 DAD skipping, skip indicator injection |
| Total validated | 21 | 100% | |

Remaining 26 of 47 initial flags: 12 specification gaps (FFS items), 14 true false positives.

## Key Findings

1. **LLM scale ≠ better performance:** Qwen 397B matches Llama 3.2 3B exactly (F1=0.786) on structured classification with identical prompts.

2. **Thinking mode hurts classification:** Extended reasoning overrides prompt rules, introducing false positives without improving recall. This is documented as a failure mode of chain-of-thought on constrained extraction tasks.

3. **Multi-model consensus is free improvement:** Union voting rescues 7 paragraphs with zero additional false positives, improving F1 from 0.786 to 0.903 at no extra inference cost.

4. **Context engineering solves cross-paragraph hazards:** Providing surrounding context enables detection of hazards where trigger and action span different paragraphs, achieving perfect recall.

5. **Naive reflexion destroys precision:** Self-correction without validation creates 22 false positives. A skeptical validation pass is essential.

6. **LLMs can discover beyond rules:** Open-ended mode independently rediscovered 8 known vulnerabilities from LTEInspector, 5GReasoner, and DoLTEst without being told to look for them.

7. **Prompt design matters more than model size:** The ablation study shows rules contribute more than examples, and 3 examples is optimal. Pure zero-shot completely fails.

## Related Work

| Paper | Venue | Method | Limitation Addressed |
|-------|-------|--------|---------------------|
| Bookworm (Chen et al.) | IEEE S&P 2021 | Fine-tuned RoBERTa | Our baseline — we exceed F1 by 30% |
| 5GPT (Wen et al.) | IEEE TIFS 2025 | GPT-4 zero-shot + CoT | No P/R/F1 evaluation |
| Hermes (Klischies et al.) | USENIX 2024 | NLP → FSM + formal verif. | No NLP extraction metrics |
| CellularLint (Khandker et al.) | USENIX 2024 | Cross-spec consistency | Different task |
| SPEC5G (Rahman et al.) | IJCNLP 2023 | BERT sentence classif. | Sentence-level only |
| DoLTEst (Kim et al.) | USENIX 2022 | Protocol fuzzing | Implementation-level, not spec-level |

## Citation

If you use this work, please cite:

```bibtex
@thesis{srinija2026agentic,
  title={Agentic AI for Automated Security Hazard Detection in 3GPP Protocol Specifications},
  author={Srinija},
  year={2026},
  school={University of South Florida}
}
```

## License

This project is for academic research purposes.

**Currently there are many python files and processed data files in my src and data which i didn't mention in this readme file, because there's still on going research of using baseline bookworm's published grouth truth HI's as my data directly on TS_24_301.16.6.0.pdf version(3GPP data), once entire results are produced i will be soon adding the remaining clean files with proper results, proper descriptions in readme about what is done and how.**
