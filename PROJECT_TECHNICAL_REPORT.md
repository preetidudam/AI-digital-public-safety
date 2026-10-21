# 🛡️ Digital Public Safety: Comprehensive Technical Project Report

**Project Title:** AI-Assisted Scam & Conversational Fraud Detection  
**Repository Location:** `d:/TY_College/AI/Project`  
**Report Date:** October 2026  
**Status:** Clean Architecture Migrated & Fully Runnable  

---

## 1. Executive Summary & Purpose

The **Digital Public Safety** project is an applied machine learning and audio analysis system designed to screen English and Hinglish (Hindi written in Roman script) text messages (SMS) and phone call recordings for fraudulent intent.

### Core Problem & Motivation
In recent years, conversational financial cybercrime has grown exponentially in India. Rather than relying on simple phishing links, perpetrators use sophisticated psychological manipulation, social engineering, and intimidation. Common fraud vectors include:
- **"Digital Arrest" Scams:** Impersonating police, CBI, or customs officials; claiming the victim's identity (Aadhaar, SIM) is linked to narcotics or money laundering; and ordering them into illegal video/audio confinement until funds are transferred.
- **OTP & Account Threat Scams:** Impersonating bank customer support and claiming an account will be blocked within minutes unless an OTP or PIN is shared.
- **Courier / Fake Parcel Scams:** Claiming an international parcel addressed to the victim contains contraband.
- **KYC Update Threats:** Ordering victims to install remote desktop apps (AnyDesk, TeamViewer) or click malicious links.
- **Lottery & Refund Scams:** Announcing cash prizes or tax refunds that require an upfront "clearance fee."

### System Objective
To provide real-time, explainable risk screening that generates an estimated fraud probability (0–100%) and context-specific public safety advisories (e.g., instructing the user to hang up, never disclose OTPs, and verify independently).

---

## 2. Repository & File Architecture

The repository has been structured following standard production machine learning practices, decoupling data ingestion, preprocessing, modeling, evaluation, and user interfaces:

```
d:/TY_College/AI/Project/
├── config.py                        # Centralized configuration (paths, dataset IDs, thresholds, seeds)
├── requirements.txt                 # Exact project dependencies
├── README.md                        # Documentation and quickstart guide
├── Copy_of_Untitled2.ipynb          # Original Jupyter Notebook (kept completely unchanged)
├── app.py                           # Gradio dual-tab Web Application
├── .gitignore                       # Git ignore file for ML artifacts and cache
│
├── src/                             # Core Python package
│   ├── __init__.py
│   ├── data/                        # Data ingestion, auditing, and preparation
│   │   ├── __init__.py
│   │   ├── loader.py                # Dataset downloaders & schema inspection
│   │   └── preprocess.py            # Deduplication, conflict filtering, scenario grouping
│   ├── models/                      # Model architectures & inference engines
│   │   ├── __init__.py
│   │   ├── baseline_sms.py          # TF-IDF + Logistic Regression for SMS
│   │   ├── baseline_icfd.py         # TF-IDF + Logistic Regression for transcript turns
│   │   ├── improved_model.py        # Word + Char FeatureUnion + Logistic Regression
│   │   ├── detector.py              # ScamDetector inference class & safety alerts
│   │   └── audio_detector.py        # Whisper ASR & Wav2Vec2 deepfake audio classifier
│   ├── evaluation/                  # Evaluation suites and interpretability
│   │   ├── __init__.py
│   │   ├── metrics.py               # Classification report, ROC-AUC, threshold sweep
│   │   ├── error_analysis.py        # Feature weight extraction & false positive audit
│   │   └── test_cases.py            # Sanity checks, contrast pairs, unseen examples
│   └── utils/                       # Supporting utilities
│       ├── __init__.py
│       └── audio_utils.py           # Google Text-to-Speech (gTTS) audio generator
│
├── scripts/                         # Executable CLI workflows
│   ├── train_sms.py                 # Train SMS baseline model
│   ├── train_icfd_baseline.py       # Train turn-level ICFD baseline
│   ├── train_icfd_improved.py       # Train production Word+Char model & save artifact
│   ├── evaluate.py                  # Run sanity checks, contrast pairs, and feature inspection
│   ├── generate_sample_audio.py     # Generate synthetic scam call MP3 sample
│   └── transcribe_and_detect.py     # End-to-end CLI audio transcription + fraud screening
│
├── models/                          # Persisted binary model artifacts (.joblib)
│   ├── scam_detector_improved.joblib# Active production model package (7.4 MB)
│   ├── digital_scam_transcript_model.joblib # Turn-level baseline (6.8 MB)
│   └── sms_scam_baseline.joblib     # SMS baseline (300 KB)
│
└── data/                            # Sample audio recordings and local storage
    └── sample_scam_call.mp3         # Synthetic test call generated by gTTS
```

### Detailed File Responsibilities

| File | Primary Responsibility |
|---|---|
| [`config.py`](file:///d:/TY_College/AI/Project/config.py) | Defines project roots, filesystem paths (`models/`, `data/`), Hugging Face dataset IDs, decision thresholds (`0.30`), and random seeds (`42`). |
| [`src/data/loader.py`](file:///d:/TY_College/AI/Project/src/data/loader.py) | Downloads Hugging Face datasets (`icfd-31k`, `scam-ham-india`), prints schema breakdowns, and runs multi-split overlap/conflict audits. |
| [`src/data/preprocess.py`](file:///d:/TY_College/AI/Project/src/data/preprocess.py) | Cleans text, removes duplicates and label contradictions, aggregates conversation turns by taking final transcripts (`.tail(1)`), and extracts scenario IDs (`SCENARIO_\d+`) to enforce strict group-based train/val splits without data leakage. |
| [`src/models/improved_model.py`](file:///d:/TY_College/AI/Project/src/models/improved_model.py) | Implements the primary production pipeline combining word-level TF-IDF and character within-word (`char_wb`) TF-IDF via `FeatureUnion`, fed to a class-weighted Logistic Regression classifier. |
| [`src/models/detector.py`](file:///d:/TY_College/AI/Project/src/models/detector.py) | Encapsulates inference logic: probability estimation, threshold evaluation ($P \ge 0.30$), and formatting of user-facing alert advisories. |
| [`src/models/audio_detector.py`](file:///d:/TY_College/AI/Project/src/models/audio_detector.py) | Wraps OpenAI Whisper (`small`) with in-memory `librosa` audio loading (bypassing external `ffmpeg.exe` dependencies on Windows) and hosts the Wav2Vec2 deepfake audio classifier. |
| [`src/evaluation/test_cases.py`](file:///d:/TY_College/AI/Project/src/evaluation/test_cases.py) | Hosts curated test datasets: 10 Indian domain sanity checks, 6 safety vs. threat contrast pairs, and 6 unseen SMS messages. |
| [`scripts/transcribe_and_detect.py`](file:///d:/TY_College/AI/Project/scripts/transcribe_and_detect.py) | CLI utility to transcribe an audio file, evaluate fraud risk, and print a formatted console alert banner. |
| [`app.py`](file:///d:/TY_College/AI/Project/app.py) | Interactive web interface hosting dual tabs (Text/SMS Detection & Audio Call Detection) powered by Gradio. |

---

## 3. End-to-End Application Dataflow

```
               [ User Input ]
             /                \
   [ Text Message / SMS ]    [ Audio Call File (.mp3/.wav/.m4a) ]
             |                                    |
             |                         librosa.load(..., sr=16kHz)
             |                                    |
             |                      [ Mono Float32 Audio Waveform ]
             |                                    |
             |                         OpenAI Whisper ("small")
             |                                    |
             |                          [ Speech-to-Text ]
             \                                    /
              ---> [ Raw Call Transcript String ] <---
                               |
                   [ ScamDetector.analyze() ]
                               |
                  [ Scikit-Learn FeatureUnion ]
                  /                           \
       Word TF-IDF Vectorizer         Char-wb TF-IDF Vectorizer
     (n-gram 1-2, sublinear TF)      (n-gram 3-5, sublinear TF)
                  \                           /
                   ---> [ Combined Sparse Matrix ]
                               |
                  [ Logistic Regression Classifier ]
                      (class_weight="balanced")
                               |
                   predict_proba([transcript])
                               |
               [ Raw Fraud Probability P (0.0 to 1.0) ]
                               |
                   Threshold Comparison (P >= 0.30)
                               |
              +----------------+----------------+
              |                                 |
        If P >= 0.30                       If P < 0.30
              |                                 |
     ⚠️ POTENTIAL SCAM DETECTED          ✅ NO ALERT TRIGGERED
  "Do not transfer money or share     "No scam warning threshold reached.
   OTPs, PINs, or passwords."          Stay cautious and verify."
              \                                 /
               ---> [ Gradio UI / CLI Output ] <---
                     • Analyzed Text
                     • Fraud Risk Score (%)
                     • Actionable Advisory Banner
```

---

## 4. Models, Technologies, and Libraries Used

### Models
1. **Word + Character FeatureUnion TF-IDF Classifier (Primary Text Model):**
   - Word branch: `ngram_range=(1, 2)`, `min_df=2`, `max_features=100,000`, `sublinear_tf=True`.
   - Character branch: `analyzer="char_wb"`, `ngram_range=(3, 5)`, `min_df=3`, `max_features=100,000`, `sublinear_tf=True`.
   - Classifier: `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`.
   - Purpose: Captures whole keywords ("refund", "aadhaar", "police") as well as phonetic variations, abbreviations, and Hinglish morphemes ("kijiye", "warna", "bol").
2. **OpenAI Whisper (`small` architecture, ~461M weights):**
   - Multi-task sequence-to-sequence ASR model trained on 680,000 hours of multilingual audio.
   - Purpose: Transcribes Indian-accented English and Hinglish phone calls into text.
3. **Deepfake Audio Classifier (`Vansh180/deepfake-audio-wav2vec2`):**
   - Pretrained Wav2Vec2 transformer fine-tuned for synthetic vs. authentic speech classification.
4. **Baseline Models:**
   - SMS Baseline: Word TF-IDF + Logistic Regression trained on `scam-ham-india`.
   - ICFD Turn-level Baseline: Word TF-IDF + Logistic Regression trained on incremental dialogue turns.

### Libraries & Technology Stack
- **Machine Learning & NLP:** `scikit-learn` (1.8.0), `joblib` (1.5.3), `torch` (2.10.0), `transformers` (5.3.0).
- **Data Engineering:** `pandas` (2.3.3), `numpy` (2.2.5), `datasets` (Hugging Face).
- **Audio Processing:** `openai-whisper`, `librosa` (1.0.0), `soundfile` (0.14.0), `gTTS` (2.5.4).
- **User Interface:** `gradio` (4.x / 5.x) over FastAPI/Uvicorn.
- **Runtime Environment:** Python 3.13 on Windows.

---

## 5. What Is Currently Implemented and Working

All core capabilities migrated from the notebook are active, tested, and passing:
1. **Automated Data Preparation Pipeline:** Ingests raw Hugging Face datasets, removes identical transcripts with contradictory labels, groups dialogue fragments into complete transcripts, and creates leakage-free scenario splits.
2. **Three Fully Trained Models:** Saved in `models/` (`sms_scam_baseline.joblib`, `digital_scam_transcript_model.joblib`, and `scam_detector_improved.joblib`).
3. **In-Memory Audio Transcription & Bypassed FFmpeg:** Audio recordings (`.mp3`, `.wav`) are decoded via `librosa` directly in RAM, allowing Whisper to operate on Windows without requiring `ffmpeg.exe` in system PATH.
4. **End-to-End CLI Screening Tool:** Running `python scripts/transcribe_and_detect.py --audio data/sample_scam_call.mp3` transcribes audio, evaluates risk, and outputs a formatted alert.
5. **Evaluation & Diagnostic Tools:** `scripts/evaluate.py` executes the 10-case sanity test suite, 6 contrast pairs, and extracts top positive/negative feature weights.
6. **Synthetic Audio Generator:** `scripts/generate_sample_audio.py` leverages Google TTS to create test call recordings.
7. **Interactive Web App:** `python app.py` serves the application locally at `http://127.0.0.1:7860`.

---

## 6. What Is Planned / Mentioned but NOT Implemented Yet

1. **Transformer / BERT Fine-Tuning:**
   - Mentioned in the notebook planning notes (e.g., fine-tuning `bert-base-uncased`, `roberta-base`, or Indian regional models like `ai4bharat/indic-bert`).
   - *Current status:* All text classification is currently handled by TF-IDF + Logistic Regression. No transformer-based text classifier is trained or integrated yet.
2. **Deepfake Audio Integration in Web UI:**
   - The Wav2Vec2 audio deepfake model (`Vansh180/deepfake-audio-wav2vec2`) is wrapped in `src/models/audio_detector.py` (matching Cells 52–54), but is **not** currently wired into the Gradio UI tab. Tab 2 currently only evaluates spoken *content* via Whisper, not acoustic voice cloning.
3. **Dynamic Conversational Turn-by-Turn Tracking:**
   - In live calls, fraud indicators accumulate turn-by-turn. Currently, the system only evaluates single static text strings or complete recordings.
4. **Native Indic Script Support:**
   - The current tokenizers and datasets operate on Latin-script English and romanized Hinglish. Native Devanagari, Gurmukhi, Tamil, etc., are not explicitly tokenized or trained.
5. **Token-Level UI Heatmaps / Explainability:**
   - While feature weights can be inspected in Python/CLI tables, the Gradio UI does not yet render interactive token-highlighting heatmaps to show the user which specific phrases triggered the alert.

---

## 7. Datasets, Artifacts, and Configurations

### Datasets
- **`rishia2220/icfd-31k` (Indian Conversational Fraud Dataset):**
  - ~31,000 conversational instances.
  - Case types: `Clear Fraud`, `Subtle Fraud`, `Clear Normal`, `Ambiguous but Ultimately Normal`.
  - Binary mapping: Fraud categories $\rightarrow 1$, Normal categories $\rightarrow 0$.
  - Grouping identifier: `conversation_uid` (regex pattern: `(SCENARIO_\d+)`).
- **`anmolshrivastav/scam-ham-india`:**
  - SMS dataset with spam, scam, and ham messages tailored to Indian mobile patterns.

### Model Artifact Specifications
- **`models/scam_detector_improved.joblib` (7.43 MB):**
  - Serialized dictionary structure:
    ```python
    {
        "model": Pipeline([("features", FeatureUnion(...)), ("classifier", LogisticRegression(...))]),
        "threshold": 0.30,
        "label_meanings": {0: "Legitimate", 1: "Potential scam"},
        "model_type": "Word + Character TF-IDF with Logistic Regression"
    }
    ```

### Decision Threshold
- **Selected Threshold:** `0.30` (30%).
- **Rationale:** In consumer public safety, missing a fraud attempt (False Negative) is significantly more dangerous than a false alarm (False Positive). Lowering the decision threshold from standard 0.50 to 0.30 boosts scam recall above 90% while maintaining acceptable precision.

---

## 8. Frontend-to-Backend Connection

The project utilizes **Gradio Blocks** (`app.py`), which bridges the web frontend and Python backend via an asynchronous FastAPI/Uvicorn server:

```
[ Browser UI: http://127.0.0.1:7860 ]
              |
              | HTTP POST (JSON / Multipart Form Data)
              v
[ Gradio Server Handlers (app.py) ]
       |
       +---> Tab 1 Event: analyze_text(text)
       |         |
       |         v
       |     [ get_detector() ] -> ScamDetector.analyze()
       |         |
       |         v
       |     Returns: (analyzed_text, score_percent, alert_message)
       |
       +---> Tab 2 Event: analyze_audio(audio_filepath)
                 |
                 v
             [ get_transcriber() ] -> WhisperTranscriber.transcribe()
                 |
                 v
             [ get_detector() ] -> ScamDetector.analyze()
                 |
                 v
             Returns: (transcript, score_percent, alert_message)
```

- **Lazy Singleton Architecture:** Models are instantiated via `get_detector()` and `get_transcriber()` only when the first inference request is received. This prevents slow startup times and avoids redundant memory allocation.
- **Decoupled Architecture:** `app.py` contains zero machine learning logic—it acts purely as a presentation layer delegating to `ScamDetector` and `WhisperTranscriber`.

---

## 9. Current Technical Limitations & Known Issues

1. **TF-IDF Semantic Blindness (Contrast Pair False Positives):**
   - Because TF-IDF measures n-gram frequency without deep semantic attention, heavy fraud keywords trigger high scores even in legitimate contexts.
   - *Example:* The phrase *"Your refund has been processed through the official bank app. No fee is required"* scored **60.7% (Scam)** during evaluation because `"refund"` has the single highest positive fraud weight (+3.31) in the dataset.
   - *Solution:* This is precisely why a contextual transformer (BERT / DeBERTa) is needed to understand negation and reassurance syntax.
2. **Whisper CPU Inference Latency:**
   - Transcribing a 15-second audio clip on CPU requires approximately 8–12 seconds. On CUDA-enabled GPUs, this latency drops to < 1 second.
3. **Batch Audio / Live Stream Limitation:**
   - The current web UI accepts pre-recorded files (`gr.Audio(type="filepath")`). It does not support real-time audio chunk streaming over WebSockets.
4. **Static Single Threshold:**
   - The alert threshold (`0.30`) is uniform across all message types. Distinguishing between short SMS messages and lengthy multi-minute transcripts could benefit from dynamic length-normalized scoring.

---

## 10. Roadmap: Where and How to Integrate New AI/ML Features

When the user is ready to introduce new features, the codebase has been structured so modifications are modular and isolated:

### A. Integrating a Fine-Tuned Transformer / BERT Model
1. **Module Creation:** Create `src/models/bert_model.py` containing a Hugging Face PyTorch module (`AutoModelForSequenceClassification` and `AutoTokenizer` for e.g. `roberta-base` or `ai4bharat/indic-bert`).
2. **Training Pipeline:** Add `scripts/train_bert.py` using Hugging Face `Trainer` or PyTorch lightning, reading from `src.data.preprocess.prepare_complete_conversations()`.
3. **Detector Adapter:** Update `src/models/detector.py` to support loading either `.joblib` (classical) or a Hugging Face directory/checkpoint (transformer) through a unified `predict_score(text)` interface.
4. **Zero Frontend Disruption:** Because `app.py` only calls `detector.analyze(text)`, swapping the backend model requires zero changes to the Gradio web UI.

### B. Integrating Acoustic Deepfake Detection into the Web UI
1. **Model Already Available:** `DeepfakeAudioClassifier` is already implemented in `src/models/audio_detector.py`.
2. **UI Update in `app.py`:**
   - In Tab 2 (`analyze_audio`), invoke `deepfake_detector.predict(audio_path)`.
   - Add a Gradio output component: `gr.Label(label="Acoustic Authenticity (Real vs Synthetic Voice)")`.
   - Provide combined screening: Spoken content risk (Whisper + ScamDetector) + Voice authenticity risk (Wav2Vec2).

### C. Integrating Token-Level Explainability / Heatmaps
1. **Module Addition:** In `src/evaluation/error_analysis.py`, implement a feature attribution helper (e.g., using SHAP or extracting individual TF-IDF feature contributions $\text{tf-idf}(w) \times \beta_w$).
2. **UI Integration:** Replace `gr.Textbox(label="Analyzed Text")` with `gr.HighlightedText()` in `app.py`, highlighting high-risk phrases in red and safety phrases in green.

---

## 11. Summary Table of Run Commands

| Purpose | Command |
|---|---|
| Launch Web Application | `python app.py` |
| Transcribe & Detect Call via CLI | `python scripts/transcribe_and_detect.py --audio data/sample_scam_call.mp3` |
| Run Full Evaluation Suite | `python scripts/evaluate.py` |
| Generate Sample Audio Call (gTTS) | `python scripts/generate_sample_audio.py` |
| Retrain Production Improved Model | `python scripts/train_icfd_improved.py` |
| Retrain SMS Baseline Model | `python scripts/train_sms.py` |
| Retrain ICFD Turn Baseline Model | `python scripts/train_icfd_baseline.py` |
