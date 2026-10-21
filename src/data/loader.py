"""
Dataset loading and inspection utilities for ICFD-31k and Scam-Ham-India datasets.
"""
from collections import Counter
from pprint import pprint
import pandas as pd
from datasets import load_dataset

from config import ICFD_DATASET_ID, SMS_DATASET_ID


def load_icfd_dataset(dataset_id: str = ICFD_DATASET_ID):
    """
    Load the Indian Conversational Fraud Dataset (ICFD-31k) from Hugging Face.
    """
    print(f"Loading ICFD dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id)
    print("ICFD dataset loaded successfully.")
    return dataset


def load_sms_dataset(dataset_id: str = SMS_DATASET_ID):
    """
    Load the Indian Scam and Legitimate SMS dataset (scam-ham-india) from Hugging Face.
    """
    print(f"Loading SMS dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id)
    print("SMS dataset loaded successfully.")
    return dataset


def inspect_dataset(dataset, name: str, num_examples: int = 3):
    """
    Inspect splits, schema, features, and sample records of a Hugging Face dataset.
    Matches Cells 3-5 of the original notebook.
    """
    print(f"\n{'=' * 50}")
    print(f"DATASET INSPECTION: {name}")
    print(f"{'=' * 50}")

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Rows: {len(split_data):,}")
        print("Columns:", split_data.column_names)
        print("Features:")
        pprint(split_data.features)

        if "label" in split_data.column_names:
            print("Label distribution:", Counter(split_data["label"]))
        if "case_type" in split_data.column_names:
            print("Case type distribution:", Counter(split_data["case_type"]))

        print(f"\nFirst {min(num_examples, len(split_data))} example(s):")
        for i in range(min(num_examples, len(split_data))):
            print(f"--- Example {i + 1} ---")
            pprint(split_data[i], sort_dicts=False)


def audit_icfd_dataset(icfd, sample_size: int = 20000):
    """
    Performs data consistency checks on the ICFD dataset across splits.
    Checks duplicate transcripts, conversation UID overlaps, and label consistency.
    Matches Cells 20-21 of the original notebook.
    """
    print("\n" + "=" * 50)
    print("ICFD DATA AUDIT")
    print("=" * 50)

    # 1. Label distribution per split
    print("\n--- Label Distribution by Split ---")
    for split_name, data in icfd.items():
        print(f"{split_name}: {Counter(data['case_type'])}")

    # 2. Conversation ID overlap across splits
    print("\n--- Conversation ID Overlap Across Splits ---")
    split_ids = {
        name: set(data.select(range(min(100_000, len(data))))["conversation_uid"])
        for name, data in icfd.items()
    }
    names = list(split_ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_ids[names[i]] & split_ids[names[j]]
            print(f"{names[i]} vs {names[j]}: {len(overlap)} IDs overlap in inspected prefix")

    # 3. Conflicting labels audit
    print(f"\n--- Conflicting Labels Audit (Sample up to {sample_size:,} per split) ---")
    frames = []
    for split_name, data in icfd.items():
        n = min(sample_size, len(data))
        sample = data.select(range(n)).to_pandas()
        frames.append(pd.DataFrame({
            "split": split_name,
            "text": sample["cumulative_text"].fillna("").astype(str).str.strip(),
            "case_type": sample["case_type"].astype(str),
            "conversation_uid": sample["conversation_uid"].astype(str)
        }))

    audit_df = pd.concat(frames, ignore_index=True)
    audit_df = audit_df[audit_df["text"].str.len() > 0]

    text_groups = audit_df.groupby("text").agg(
        rows=("case_type", "size"),
        label_count=("case_type", "nunique"),
        labels=("case_type", lambda x: sorted(set(x))),
        splits=("split", lambda x: sorted(set(x)))
    ).reset_index()

    print(f"Sampled rows: {len(audit_df):,}")
    print(f"Unique exact transcripts: {len(text_groups):,}")
    conflicts_count = int((text_groups["label_count"] > 1).sum())
    print(f"Transcripts with conflicting labels: {conflicts_count:,}")

    return text_groups
