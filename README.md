# 🛡️ Digital Public Safety: AI-Assisted Scam & Conversational Fraud Detection

A machine learning and audio analysis system designed to screen English and Hinglish messages (SMS) and phone call transcripts/recordings for scam and fraud intent, with a focus on prevalent fraud vectors in India (Digital Arrest scams, OTP/KYC theft, courier/police impersonation, and lottery scams).

Migrated directly from `Copy_of_Untitled2.ipynb` into a clean, modular Python codebase runnable outside Google Colab.

---

## 📁 Project Structure

```
d:/TY_College/AI/Project/
├── Copy_of_Untitled2.ipynb          # Original Google Colab notebook (kept unchanged)
├── README.md                        # Project documentation and guide
├── requirements.txt                 # Project dependencies
├── config.py                        # Central configuration (paths, dataset IDs, thresholds, seeds)
├── .gitignore                       # Standard ML/Python gitignore
│
├── src/                             # Core Python source package
│   ├── __init__.py
│   ├── data/                        # Data ingestion, auditing, and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py                # Hugging Face dataset loaders & inspection routines
│   │   └── preprocess.py            # Deduplication, conversation aggregation, scenario grouping
│   ├── models/                      # ML pipelines and inference wrappers
│   │   ├── __init__.py
│   │   ├── baseline_sms.py          # TF-IDF + Logistic Regression for SMS
│   │   ├── baseline_icfd.py         # TF-IDF + Logistic Regression for ICFD transcripts
│   │   ├── improved_model.py        # Word + Char FeatureUnion + Logistic Regression
│   │   ├── detector.py              # ScamDetector inference engine & alert generator
│   │   └── audio_detector.py        # Whisper ASR transcription & deepfake audio classifier
│   ├── evaluation/                  # Metrics, diagnostic suites, and test cases
│   │   ├── __init__.py
│   │   ├── metrics.py               # Classification report, ROC-AUC, threshold sweep
│   │   ├── error_analysis.py        # Top feature weights, false positive/negative analysis
│   │   └── test_cases.py            # Sanity-check suite, contrast pairs, unseen examples
│   └── utils/                       # Utility scripts & helpers
│       ├── __init__.py
│       └── audio_utils.py           # Synthetic scam call generator via gTTS
│
├── scripts/                         # Executable CLI workflows
│   ├── train_sms.py                 # Train & evaluate SMS baseline model
│   ├── train_icfd_baseline.py       # Train & evaluate ICFD turn-level baseline model
│   ├── train_icfd_improved.py       # Train improved Word+Char model & save artifact
│   ├── evaluate.py                  # Run sanity checks, contrast pairs, and feature inspection
│   ├── generate_sample_audio.py     # Generate synthetic scam call audio sample (gTTS)
│   └── transcribe_and_detect.py     # CLI pipeline to transcribe audio & detect scam
│
├── app.py                           # Interactive Gradio Web App (SMS & Audio Detection)
├── models/                          # Saved model artifacts (.joblib)
└── data/                            # Sample audio recordings and local data cache
```

---

## 🗺️ Migration Mapping (Notebook Cells to Modular Files)

| Notebook Section | Original Cells | Modular Implementation |
|---|---|---|
| Dataset Loading & Inspection | Cells 0 – 8, 15 – 21 | [`src/data/loader.py`](file:///d:/TY_College/AI/Project/src/data/loader.py) |
| SMS Data Cleaning & Train/Test Split | Cells 9 – 10 | [`src/data/preprocess.py`](file:///d:/TY_College/AI/Project/src/data/preprocess.py) (`clean_sms_data`, `split_sms_data`) |
| SMS Baseline Pipeline & Eval | Cells 11 – 13 | [`src/models/baseline_sms.py`](file:///d:/TY_College/AI/Project/src/models/baseline_sms.py), [`scripts/train_sms.py`](file:///d:/TY_College/AI/Project/scripts/train_sms.py) |
| Unseen SMS Examples Test | Cell 14 | [`src/evaluation/test_cases.py`](file:///d:/TY_College/AI/Project/src/evaluation/test_cases.py) (`SMS_UNSEEN_TEST_CASES`) |
| ICFD Cleaning, Conflict Removal & Grouped Split | Cells 22 – 23 | [`src/data/preprocess.py`](file:///d:/TY_College/AI/Project/src/data/preprocess.py) (`clean_icfd_training_data`, `split_icfd_grouped`) |
| ICFD Turn-level Baseline & Eval | Cells 24 – 26 | [`src/models/baseline_icfd.py`](file:///d:/TY_College/AI/Project/src/models/baseline_icfd.py), [`scripts/train_icfd_baseline.py`](file:///d:/TY_College/AI/Project/scripts/train_icfd_baseline.py) |
| Interpretability & Top Features | Cells 27 – 28, 41 | [`src/evaluation/error_analysis.py`](file:///d:/TY_College/AI/Project/src/evaluation/error_analysis.py) (`get_top_features`) |
| Error Analysis & Length Checks | Cells 29 – 30 | [`src/evaluation/error_analysis.py`](file:///d:/TY_College/AI/Project/src/evaluation/error_analysis.py) (`analyze_prediction_errors`) |
| Conversation Aggregation & Scenario Group Split | Cells 31 – 33 | [`src/data/preprocess.py`](file:///d:/TY_College/AI/Project/src/data/preprocess.py) (`prepare_complete_conversations`, `split_by_scenario`) |
| Scenario Baseline & Threshold Sweep | Cells 34 – 36 | [`src/models/baseline_icfd.py`](file:///d:/TY_College/AI/Project/src/models/baseline_icfd.py), [`src/evaluation/metrics.py`](file:///d:/TY_College/AI/Project/src/evaluation/metrics.py) |
| Inference & Model Packaging | Cells 36 – 37, 39 | [`src/models/detector.py`](file:///d:/TY_College/AI/Project/src/models/detector.py) (`ScamDetector`) |
| 10 Sanity Checks Test Suite | Cell 40 | [`src/evaluation/test_cases.py`](file:///d:/TY_College/AI/Project/src/evaluation/test_cases.py) (`SANITY_CHECK_TEST_CASES`) |
| Contrast Pairs Test Suite | Cell 42 | [`src/evaluation/test_cases.py`](file:///d:/TY_College/AI/Project/src/evaluation/test_cases.py) (`CONTRAST_TEST_PAIRS`) |
| False Positives Scenario Audit | Cells 43 – 46 | [`src/evaluation/error_analysis.py`](file:///d:/TY_College/AI/Project/src/evaluation/error_analysis.py) (`analyze_false_positives_by_scenario`) |
| Improved Word + Char FeatureUnion Model | Cells 47 – 50 | [`src/models/improved_model.py`](file:///d:/TY_College/AI/Project/src/models/improved_model.py), [`scripts/train_icfd_improved.py`](file:///d:/TY_College/AI/Project/scripts/train_icfd_improved.py) |
| Text-only Gradio Interface | Cell 51 | Available via `app.py` or modular detector |
| Deepfake Audio Classification (Wav2Vec2) | Cells 52 – 54 | [`src/models/audio_detector.py`](file:///d:/TY_College/AI/Project/src/models/audio_detector.py) (`DeepfakeAudioClassifier`) |
| Whisper Speech-to-Text Transcription | Cells 55 – 57, 59 | [`src/models/audio_detector.py`](file:///d:/TY_College/AI/Project/src/models/audio_detector.py) (`WhisperTranscriber`) |
| Synthetic Audio Generation (gTTS) | Cells 58, 66 | [`src/utils/audio_utils.py`](file:///d:/TY_College/AI/Project/src/utils/audio_utils.py), [`scripts/generate_sample_audio.py`](file:///d:/TY_College/AI/Project/scripts/generate_sample_audio.py) |
| Audio Transcript Analysis Pipeline | Cells 60, 67 – 69 | [`scripts/transcribe_and_detect.py`](file:///d:/TY_College/AI/Project/scripts/transcribe_and_detect.py) |
| Full Gradio Web Application (Text + Audio Tabs) | Cell 70 | [`app.py`](file:///d:/TY_College/AI/Project/app.py) |

---

## 🚀 Setup & Installation

### 1. Requirements
Ensure Python 3.10+ is installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch, scikit-learn, pandas, librosa, and transformers can also leverage GPU/CUDA if available on your machine).*

---

## 🛠️ Usage & Workflows

### 1. Train the Improved Scam Detection Model
Trains the Word + Character TF-IDF FeatureUnion with class-weighted Logistic Regression on full conversation transcripts grouped by scenario, performs threshold sweeps, and saves the production model package to `models/scam_detector_improved.joblib`:

```bash
python scripts/train_icfd_improved.py
```

### 2. Run Comprehensive Model Evaluation
Evaluates the saved model against 10 domain sanity checks, safety vs. fraud contrast pairs, and displays top fraud-associated n-grams:

```bash
python scripts/evaluate.py
```

### 3. Generate a Sample Audio Call
Creates a synthetic scam call MP3 using Google Text-to-Speech (gTTS) for testing:

```bash
python scripts/generate_sample_audio.py
```

### 4. Transcribe Audio and Run Scam Detection via CLI
Transcribes an audio file with OpenAI Whisper and evaluates the transcript with the scam detector:

```bash
python scripts/transcribe_and_detect.py --audio data/sample_scam_call.mp3
```

### 5. Launch the Interactive Gradio Web App
Starts the dual-tab web application for text analysis and audio recording transcription + fraud detection:

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 📊 Datasets Used
- **ICFD-31k**: Indian Conversational Fraud Dataset (`rishia2220/icfd-31k`) containing authentic and subtle conversational scam transcripts and normal calls.
- **Scam-Ham-India**: Indian context scam and legitimate SMS dataset (`anmolshrivastav/scam-ham-india`).
