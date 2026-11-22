# Before You Submit: Fill-in and Verify Checklist

Files in `output/`:
- `proposal_document.docx` (12 pages, A4)
- `project_presentation.pptx` (17 slides, built on the course template, speaker notes on every slide)
- `oral_defense_qna.md`

Search each file for **`[FILL`** and **`[VERIFY`**. Every item is listed below.

## 1. Information only you can fill in  `[FILL: ...]`

| Where | What to fill in |
|---|---|
| Proposal: cover page, team table (4 rows) | Name, Roll No., Division for each member |
| Proposal: cover page + Section 1 | Faculty guide: Name, Designation, Department |
| Proposal: cover page | Institute / Department name |
| Slide 1 (Title) | Member names, guide name (one line under the subtitle) |
| Slide 2 (Team details) | Team table (4 rows); guide; institute |
| Slide 2 speaker notes | Each member's role (e.g. data/models, speech, currency, app) |
| Slide 9 (Collaborations) | Any confirmed partner; confirm the NSS awareness-drive idea or delete it |
| Slide 9 speaker notes | Update if you have any real collaboration |

If the team has a fifth member, add a row in both the proposal and slide 2.

## 2. Details to check against the original source  `[VERIFY]`

| # | Item | What to check |
|---|---|---|
| 1 | Reference [1]: MHA cyber-fraud loss ₹22,845.73 crore (2024) | Lok Sabha question number and date (reported July 2025). Check sansad.in or PIB. |
| 2 | Reference [2]: digital-arrest 1,23,672 cases, ₹1,935.51 crore (2024) | Which House, question number and date. One outlet (ThePrint) gives ₹1,918 crore, so cite the official reply. |
| 3 | Reference [3]: RBI Annual Report 2024-25, fake ₹500 notes +37.3% | Confirm in the "Currency Management" chapter. The RBI's 2025-26 report (May 2026) also exists; you may prefer the newer figure. |
| 4 | Reference [7]: TeleAntiFraud-28k | Only the arXiv version (2503.24115) is cited. Check whether it was later published at a conference. |
| 5 | Reference [9] + literature table row 6: Springer 2022 currency paper (DOI 10.1007/978-981-19-0840-8_62) | **Author names are missing.** Look them up on SpringerLink and also check the evaluation protocol claim in the "Limitation" column. |
| 6 | Literature table row 3: Derakhshan et al. (ASsET) | Exact accuracy figure (the abstract only says "high accuracy"). |
| 7 | Proposal Section 4.2: DPDP Act section numbers (5–6 notice/consent, 8 Data Fiduciary obligations) | Quick check against the Act text. |
| 8 | Budget: Colab Pro at about ₹1,000/month | Current Indian price. If it differs, the totals change. Tell me and I'll regenerate both files so they stay identical. |

## 3. Claims that are PLANNED, not implemented (don't overclaim)

Both documents label these **[Proposed]**. In the defense, say "we propose" or "planned", never "we built":

- DPDP redaction (Aadhaar/PAN/phone/UPI/OTP masking), notice and consent flow, auto-deletion of uploads
- Phrase-level explanation highlights; Grad-CAM heat-maps for currency
- Distilled HingBERT / IndicBERT student; INT8 quantisation; ONNX / TFLite export
- On-device inference and on-device speech recognition. **Today the phone app calls the laptop server; the models do not run on the phone.**
- Live, turn-by-turn call analysis
- Devanagari / Marathi script support
- Voice-deepfake detection (Wav2Vec2 code exists but isn't connected to the app or evaluated)
- Any collaboration or MoU with banks, CSCs, cyber cell or NSS (none in place)
- The Phase-3 target of currency hold-out macro-F1 ≥ 0.85 and the "<60 ms on phone" target

## 4. Known weaknesses an examiner may find (prepared answers are in the Q&A)

- **Digital-arrest spot checks score only 20–37%** (near or below the thresholds). This is the most important weakness given the title. See Q11.
- **Thresholds were tuned on the same validation set** they are reported on. See Q6.
- **ICFD-31k is synthetic**, so there is no real-call evaluation yet.
- **Currency hold-out:** only 47 of 90 fakes caught (macro-F1 0.712).
- **Whisper takes 3.7 s** for a 13.6 s clip on a CPU and misses the 60 ms target.

## 5. Corrections to the old project report

`PROJECT_TECHNICAL_REPORT.md` contains numbers that our measurements contradict. **Don't quote it in the defense.** The new documents use the measured values:

| Old report says | Measured / correct |
|---|---|
| Threshold 0.30 gives "scam recall above 90%" | Recall is 0.786 at 0.30; 0.896 only at 0.20 |
| Whisper small "~461M weights" | 241 M parameters |
| Whisper takes 8–12 s for 15 s audio on CPU | 3.7 s for a 13.6 s clip on this laptop |
| "refund … no fee required" scored 60.7% | Scores 79.7% with the current model |

## 6. Demo-day tip

The text detector gave a pasted digital-arrest paragraph only 20–37% during testing, so in balanced mode it may say "no alert" live. For the demo, either:
- use the bank OTP example in `message.txt` (scores 68%), or
- explain the high-recall mode and the limitation before showing a digital-arrest example.
