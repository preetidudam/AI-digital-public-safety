"""
Data preprocessing, cleaning, and dataset splitting pipelines.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from config import SMS_LABEL_MAP, ICFD_LABEL_MAP, RANDOM_SEED


def clean_sms_data(sms_split_or_df) -> pd.DataFrame:
    """
    Clean and normalize the SMS scam/ham dataset.
    Matches Cells 9-10 of the original notebook.
    """
    if hasattr(sms_split_or_df, "to_pandas"):
        sms_df = sms_split_or_df.to_pandas()
    else:
        sms_df = sms_split_or_df.copy()

    assert {"text", "label"}.issubset(sms_df.columns), (
        f"Unexpected columns: {sms_df.columns.tolist()}"
    )

    # Keep text and label, then clean
    sms_df = sms_df[["text", "label"]].copy()
    sms_df["text"] = sms_df["text"].astype("string").str.strip()
    sms_df["label"] = sms_df["label"].astype("string").str.strip().str.lower()

    sms_df = sms_df.dropna()
    sms_df = sms_df[sms_df["text"].str.len() > 0]
    sms_df = sms_df.drop_duplicates(subset=["text"])

    # Map labels to numeric targets
    sms_df["target"] = sms_df["label"].map(SMS_LABEL_MAP)

    if sms_df["target"].isna().any():
        unknown = sms_df.loc[sms_df["target"].isna(), "label"].unique()
        raise ValueError(f"Unmapped labels found in SMS dataset: {unknown}")

    sms_df["target"] = sms_df["target"].astype(int)
    return sms_df


def split_sms_data(
    sms_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified train-test split on the SMS dataset.
    Matches Cell 10 of the original notebook.
    """
    X = sms_df["text"].astype(str)
    y = sms_df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def clean_icfd_training_data(icfd_train_split) -> pd.DataFrame:
    """
    Clean ICFD training split, map fraud categories to binary labels,
    and remove transcripts with conflicting labels.
    Matches Cell 22 of the original notebook.
    """
    if hasattr(icfd_train_split, "to_pandas"):
        train_df = icfd_train_split.to_pandas()
    else:
        train_df = icfd_train_split.copy()

    required_cols = ["cumulative_text", "case_type", "conversation_uid"]
    train_df = train_df[required_cols].copy()

    train_df["text"] = train_df["cumulative_text"].fillna("").astype(str).str.strip()
    train_df["case_type"] = train_df["case_type"].astype(str)

    # Map categories to binary labels
    train_df["label"] = train_df["case_type"].map(ICFD_LABEL_MAP)

    # Remove empty text and unknown categories
    train_df = train_df[
        train_df["text"].ne("") & train_df["label"].notna()
    ].copy()

    # Identify and remove texts with conflicting labels
    conflicts = train_df.groupby("text")["label"].nunique()
    conflict_texts = conflicts[conflicts > 1].index

    train_df = train_df[~train_df["text"].isin(conflict_texts)]
    train_df = train_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    train_df["label"] = train_df["label"].astype(int)

    return train_df


def split_icfd_grouped(
    train_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split ICFD dataset grouped by conversation_uid to prevent leakage.
    Matches Cell 23 of the original notebook.
    """
    groups = train_df["conversation_uid"].astype(str)
    X = train_df["text"]
    y = train_df["label"].astype(int)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    return X_train, X_val, y_train, y_val, groups


def prepare_complete_conversations(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate conversation turns by taking the final/longest transcript turn per conversation,
    and extract scenario groups (SCENARIO_<id>) from conversation_uid.
    Matches Cells 32-33 of the original notebook.
    """
    conversation_df = (
        train_df.sort_values("text", key=lambda s: s.str.len())
        .groupby("conversation_uid", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    conversation_df = conversation_df[
        ["conversation_uid", "text", "label", "case_type"]
    ].copy()

    # Extract scenario ID from conversation_uid
    conversation_df["scenario_group"] = (
        conversation_df["conversation_uid"]
        .str.extract(r"(SCENARIO_\d+)", expand=False)
    )

    assert conversation_df["scenario_group"].notna().all(), (
        "Some IDs do not contain SCENARIO_<number>. Inspect those IDs first."
    )

    return conversation_df


def split_by_scenario(
    conversation_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split conversations grouped by scenario_group so that scenarios in validation
    are completely unseen during training.
    Matches Cell 33 of the original notebook.
    """
    X = conversation_df["text"].astype(str)
    y = conversation_df["label"].astype(int)
    groups = conversation_df["scenario_group"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    return X_train, X_val, y_train, y_val, groups
