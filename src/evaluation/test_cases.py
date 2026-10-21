"""
Standard evaluation test suites, contrast pairs, and sanity-check datasets.
Matches Cells 14, 26, 40, and 42 of the original notebook.
"""

# Cell 14: Unseen test messages for SMS model
SMS_UNSEEN_TEST_CASES = [
    # Scam examples
    ("CBI Arrest Threat", "A caller claiming to be from CBI threatened to arrest me unless I transferred money.", 1),
    ("Bank OTP Block Threat", "Your bank account will be blocked in five minutes. Tell me the OTP now.", 1),
    ("Aadhaar Crime Video Call", "Someone said my Aadhaar was linked to a crime and ordered me to stay on a video call.", 1),
    # Legitimate examples
    ("Official App Verification", "My bank asked me to check my debit card application in its official app.", 0),
    ("Police Public Awareness Warning", "The police issued a public warning about digital arrest scams. Never transfer money to callers.", 0),
    ("College Examination Notice", "My college called to confirm tomorrow's examination timetable.", 0),
]

# Cell 26: ICFD baseline test cases
ICFD_BASELINE_TEST_CASES = [
    (
        "Digital arrest scam",
        "A caller claimed to be from the CBI, threatened to arrest me, "
        "ordered me to stay on a video call, and demanded money."
    ),
    (
        "OTP scam",
        "Someone claiming to be from my bank demanded my OTP immediately "
        "and threatened to block my account."
    ),
    (
        "Legitimate bank call",
        "My bank asked me to check my debit card application using its "
        "official mobile banking app. They did not ask for money or an OTP."
    ),
    (
        "Public awareness message",
        "Police warned citizens that scammers may impersonate CBI officers "
        "and demand money. Citizens should report suspicious calls."
    ),
    (
        "Subtle suspicious call",
        "Someone called about a problem with my identity documents and "
        "told me to keep it secret from my family and cooperate to avoid "
        "legal action."
    )
]

# Cell 40: 10 Comprehensive domain sanity checks
SANITY_CHECK_TEST_CASES = [
    (
        "Digital arrest scam", 1,
        "Police officer calling about your Aadhaar. You are involved "
        "in a criminal case. Transfer money for verification or be arrested."
    ),
    (
        "OTP theft", 1,
        "I am from bank support. Tell me the OTP you received to unblock "
        "your account or it will be frozen."
    ),
    (
        "Fake parcel threat", 1,
        "Your parcel contains illegal items. Transfer money now to clear "
        "your name or you will be arrested."
    ),
    (
        "Legitimate bank call", 0,
        "I am calling from your bank about your service request. "
        "Never share your OTP, PIN or password. Verify using the official number."
    ),
    (
        "Legitimate delivery", 0,
        "Your parcel is out for delivery today. Check tracking in the "
        "official delivery app. No payment is required on this call."
    ),
    (
        "Suspicious KYC request", 1,
        "Your KYC expires today. Install this remote access application "
        "and tell me the code on your screen."
    ),
    (
        "Hinglish digital arrest", 1,
        "Main cyber crime department se bol raha hoon. Aapke Aadhaar par "
        "case hai. Paise transfer kijiye warna arrest hoga."
    ),
    (
        "Hinglish legitimate bank", 0,
        "Namaste sir, main bank customer service se bol raha hoon. "
        "Aapki service request ka status batana tha. OTP ya PIN mat batayiye."
    ),
    (
        "Ambiguous opening", 0,
        "Hello sir, main aapko aapke account ke baare mein call kar raha hoon. "
        "Kya abhi baat kar sakte hain?"
    ),
    (
        "Prize scam", 1,
        "Congratulations, you won a cash prize. Pay a processing fee "
        "and send your bank details to release the money."
    )
]

# Cell 42: Contrast pairs test
CONTRAST_TEST_PAIRS = [
    {
        "Pair": "Bank OTP",
        "Version": "Unsafe request",
        "Expected": "Scam",
        "Expected_binary": 1,
        "Text": "I am from your bank. Tell me your OTP and PIN "
                "immediately or your account will be blocked."
    },
    {
        "Pair": "Bank OTP",
        "Version": "Safety advice",
        "Expected": "Legitimate",
        "Expected_binary": 0,
        "Text": "I am calling from your bank to remind you never "
                "to share your OTP or PIN with anyone."
    },
    {
        "Pair": "Refund",
        "Version": "Fee demanded",
        "Expected": "Scam",
        "Expected_binary": 1,
        "Text": "Your refund is approved. Pay a processing fee "
                "to my personal UPI ID to receive the money."
    },
    {
        "Pair": "Refund",
        "Version": "Normal status update",
        "Expected": "Legitimate",
        "Expected_binary": 0,
        "Text": "Your refund has been processed. Check its status "
                "through the official bank application. No fee is required."
    },
    {
        "Pair": "Police",
        "Version": "Threat and payment",
        "Expected": "Scam",
        "Expected_binary": 1,
        "Text": "I am a police officer. Transfer money to settle "
                "this case immediately. Do not tell your family."
    },
    {
        "Pair": "Police",
        "Version": "Independent verification advice",
        "Expected": "Legitimate",
        "Expected_binary": 0,
        "Text": "Someone claiming to be a police officer called me. "
                "I will verify the claim using the official police contact."
    }
]
