# Oral Defense: 15 Likely Examiner Questions

**Project:** Explainable Lightweight Hinglish Scam-Call and SMS Screening for Digital-Arrest Fraud on Low-Cost Smartphones (Scam Shield)

Every number below matches the proposal and the slides. "Implemented" means built and measured today. "Proposed" means planned work. Don't present proposed work as if it already exists.

---

## Technical

**1. Why TF-IDF with logistic regression instead of a transformer like BERT?**
We had to fit a ₹8,000 phone: under 30 MB and under 60 ms. Our model is 7.4 MB and scores a message in 2.0 ms on a CPU. RoBERTa-base is about 125 M parameters (~500 MB) and needs a GPU server. The linear model is also transparent: every word and character pattern has a weight we can show the user. Its weakness is meaning. It can't understand negation, so we propose a distilled HingBERT student under 30 MB (INT8) as the second stage, with the n-gram model kept as the fast first filter.

**2. What do the character n-grams add?**
Hinglish has no fixed spelling ("kijiye" / "kijie", "warna" / "varna", "turant"). Character 3–5-grams inside words match these variants and abbreviations that a word-only model treats as unrelated tokens. On the same scenario-disjoint split, macro-F1 rose from 0.756 (word-only baseline) to 0.805 (word + char, threshold 0.30).

**3. How did you prevent data leakage?**
ICFD-31k has 712,316 streaming chunks. We merged chunks into full conversations, removed transcripts whose labels contradicted each other, and split **by scenario ID**, so validation scenarios never appear in training. An earlier turn-level baseline split by conversation reached macro-F1 0.971. But the same scenario templates appear on both sides of that split, so we don't use it as our headline result. For currency images we split by base-image cluster and also kept an external test set from an unseen source.

**4. Why macro-F1, and why two thresholds?**
Macro-F1 averages the F1 of both classes, so a model can't look good by favouring the larger class. The threshold trades false alarms against missed scams:
- **Balanced mode, 0.30:** scam precision 0.882, recall 0.786, macro-F1 0.805.
- **High-recall mode, 0.20:** precision 0.847, recall 0.896, macro-F1 0.831.

For digital-arrest screening a missed scam costs a victim their savings, so we recommend high-recall mode there. *We do not claim over 90% recall at 0.30.*

**5. The ICFD-31k paper reports F1 99.40. Why is yours only 0.805?**
The settings aren't comparable:
- Their 99.40 is in-domain with a ~500 MB RoBERTa. On unseen scam types they report 92.97.
- Our 0.805 is a 7.4 MB linear model on a scenario-disjoint split, which is a harder, unseen-scenario test.

Our contribution is deployability and transparency, not beating RoBERTa on accuracy. Closing that gap with a small distilled student is exactly the proposed research.

**6. Are your thresholds tuned on the test set?**
Yes, and that's a known limitation. The thresholds were chosen on the same validation set the numbers are reported on. A separate, untouched test split is part of Phase 3. Say this openly if asked.

## Novelty

**7. What exactly is new in your work?**
Existing Indian conversational-fraud work gets high F1 with ~125 M-parameter server models. We show that a 7.4 MB model screens Hinglish call transcripts in about 2 ms on a CPU (macro-F1 0.805 on unseen scenarios, roughly 65× smaller than RoBERTa-base). We then propose to:
- close its meaning gap with a distilled HingBERT student under 30 MB (INT8);
- wrap it with on-device DPDP redaction;
- add phrase-level explanations.

The goal is alerts that are private and understandable on a low-cost phone. The combination is the novelty. The redaction, explanations and student are **proposed**.

**8. Why not just use ChatGPT or Google's APIs?**
- They need a reliable internet connection.
- They cost money per request.
- They are slow for a live call.
- They send the full conversation, including Aadhaar or account numbers, to a foreign server, which conflicts with DPDP data-minimisation.

Google Vision isn't trained to tell genuine Indian notes from counterfeits, and none of these tools explain their answer in Hinglish.

## Deployment

**9. Does it run on the phone today?**
Partly. Today it's an **installable Android app (PWA)** that talks to our Python server through one HTTPS link. The models run on the server or laptop, not on the phone. Moving inference on-device (ONNX / TFLite INT8 export, on-device speech recognition) is **proposed** Phase 4 work. The text and currency models are already small enough (7.4 MB / 2 ms and 6.2 MB / 13 ms on a laptop CPU), but on-phone latency hasn't been measured yet.

**10. Whisper takes 3.7 seconds. How can this work during a live call?**
It can't yet, and we say so. Whisper-small (241 M parameters) took 3.7 s to transcribe a 13.6 s clip on our laptop CPU, which misses the 60 ms target and would be slower on a phone. Today the app works on recordings or pasted text. Live, turn-by-turn analysis and a smaller on-device speech model are future work. The 2 ms text stage itself is fast enough to run after every sentence once speech-to-text is fast.

**11. Your title targets digital-arrest fraud, but your spot checks score those scripts at only 20–37%. Isn't that a failure?**
It's the most important limitation we found, and we report it. Short, single-paragraph digital-arrest scripts land near the threshold because the model was trained on long, multi-turn synthetic dialogues and relies on keywords. A bank OTP scam scored 68% and a genuine service call 1%. Our responses:
1. Use high-recall mode (0.20) for this use case.
2. Collect real, consented digital-arrest examples in the pilot.
3. Add the distilled HingBERT student, which reads context instead of counting keywords.

These spot checks are illustrations, not a test set.

## Data privacy

**12. How do you comply with the DPDP Act, 2023?**
- **Training data:** public, and the conversation data (ICFD-31k) is synthetic, so no real person's call was used.
- **Prototype:** doesn't save inputs to a database, but the web framework keeps uploaded files in a temporary cache. Automatic deletion is planned.
- **Proposed, before deployment:**
  1. On-device redaction of Aadhaar (12 digits with Verhoeff checksum), PAN, phone, account/IFSC, UPI IDs and OTPs before scoring or storage.
  2. Clear notice and opt-in consent before recording audio (Sections 5–6).
  3. Use data only for the screening request, then delete it. No training on user data without separate consent (Section 8).

**13. Can it detect AI-cloned (deepfake) voices?**
Not in the app today. The analysis checks *what is said*, not whether the voice is synthetic. A Wav2Vec2 deepfake-audio classifier wrapper exists in our code but isn't connected to the user interface or evaluated, so we don't claim it.

## Currency module

**14. Your currency model catches only 52% of fakes on the hold-out set. Why include it?**
It's a secondary module, and we present it as a research gap, not a finished product. Macro-F1 is 0.916 on validation and 0.852 on an external source, but 0.712 on the final hold-out set: only 47 of 90 counterfeit notes were caught. This shows that photo-based detection doesn't transfer well across camera and image sources, which published work rarely tests. The app labels it a "second opinion", not a certified test. The Phase 3 target is hold-out macro-F1 ≥ 0.85 using phone-captured data and domain-generalisation training. It can never replace physical checks (intaglio print, watermark, UV).

## Budget

**15. Why ₹4.02 lakh, and why is manpower the largest head?**

| Year | Purpose | Amount |
|---|---|---|
| Year 1 | 12-month research | ₹1,53,180 |
| Year 2 | District pilot | ₹1,73,305 |
| Year 3 | Scale-up | ₹75,900 |
| **Total** | | **₹4,02,385** |

Compute is cheap because the models are small: Colab Pro ₹12,000 and hosting ₹45,000 over 30 months. The real costs are people: Hinglish annotation (₹80,000), a pilot field assistant (₹60,000) and a maintenance developer (₹30,000), so manpower comes to ₹1,70,000. Contingency is 5% and institutional overheads 10% of direct costs. Every row and column sums exactly. If funding is cut, the pilot field assistant is the first item to scale down, since annotation drives model quality.

---

## Quick reference: numbers to remember

| Item | Value |
|---|---|
| Cyber-fraud loss, 2024 (MHA) | ₹22,845.73 crore (₹7,465.18 crore in 2023) |
| Digital-arrest cases, 2024 (MHA) | 1,23,672 cases, ₹1,935.51 crore lost |
| Fake ₹500 notes, 2024-25 (RBI) | +37.3% (1,17,722 vs 85,711) |
| Text model, balanced (0.30) | P 0.882 / R 0.786 / macro-F1 0.805 |
| Text model, high-recall (0.20) | P 0.847 / R 0.896 / macro-F1 0.831 |
| Word-only baseline | macro-F1 0.756 |
| SMS baseline | F1 0.99 (n = 455) |
| Text model size / speed | 7.4 MB / 2.0 ms |
| Currency CNN val / external / hold-out | 0.916 / 0.852 / 0.712 macro-F1 |
| Currency CNN size / speed | 6.2 MB / 13 ms, 1.52 M parameters |
| Whisper-small | 241 M parameters, 3.7 s for a 13.6 s clip (CPU) |
| PERT critical path | A→B→C→F→H→J→K = 51 weeks |
| Budget | ₹1,53,180 + ₹1,73,305 + ₹75,900 = ₹4,02,385 |
