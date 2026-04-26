import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


IDENTITY_DASH_PATTERN = re.compile(r"^id-\d+$", re.IGNORECASE)

CORE_TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
]

CORE_IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


@dataclass
class TemporalSplit:
    train_x: pd.DataFrame
    train_y: pd.Series
    val_x: pd.DataFrame
    val_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def normalize_identity_column_name(column_name: str) -> str:
    if IDENTITY_DASH_PATTERN.match(column_name):
        return column_name.replace("-", "_")
    return column_name


def normalize_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column_name: normalize_identity_column_name(column_name)
        for column_name in frame.columns
    }
    return frame.rename(columns=rename_map)


def _resolve_use_columns(csv_path: str, requested_columns: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not requested_columns:
        return None

    header_frame = pd.read_csv(csv_path, nrows=0)
    available_columns = set(header_frame.columns)
    resolved_columns = [column_name for column_name in requested_columns if column_name in available_columns]

    if "TransactionID" in available_columns and "TransactionID" not in resolved_columns:
        resolved_columns.insert(0, "TransactionID")

    return resolved_columns or None


def _read_csv(
    csv_path: str,
    use_columns: Optional[List[str]],
    chunksize: Optional[int],
    max_rows: Optional[int],
) -> pd.DataFrame:
    if max_rows and max_rows > 0:
        return pd.read_csv(csv_path, usecols=use_columns, nrows=max_rows, low_memory=False)

    if chunksize and chunksize > 0:
        frames = []
        for chunk_frame in pd.read_csv(
            csv_path,
            usecols=use_columns,
            chunksize=chunksize,
            low_memory=False,
        ):
            frames.append(chunk_frame)
        if not frames:
            return pd.DataFrame(columns=use_columns or [])
        return pd.concat(frames, ignore_index=True)

    return pd.read_csv(csv_path, usecols=use_columns, low_memory=False)


def _split_paths(data_dir: str, split: str) -> Tuple[str, str]:
    transaction_csv = os.path.join(data_dir, f"{split}_transaction.csv")
    identity_csv = os.path.join(data_dir, f"{split}_identity.csv")

    if not os.path.exists(transaction_csv):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {transaction_csv}")
    if not os.path.exists(identity_csv):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {identity_csv}")

    return transaction_csv, identity_csv


def load_ieee_split(
    data_dir: str,
    split: str,
    transaction_columns: Optional[Iterable[str]] = None,
    identity_columns: Optional[Iterable[str]] = None,
    chunksize: Optional[int] = 100_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split chỉ chấp nhận 'train' hoặc 'test'.")

    transaction_csv, identity_csv = _split_paths(data_dir, split)

    resolved_transaction_columns = _resolve_use_columns(transaction_csv, transaction_columns)
    resolved_identity_columns = _resolve_use_columns(identity_csv, identity_columns)

    transaction_frame = _read_csv(transaction_csv, resolved_transaction_columns, chunksize, max_rows)
    identity_frame = _read_csv(identity_csv, resolved_identity_columns, chunksize, max_rows)
    identity_frame = normalize_identity_columns(identity_frame)

    if "TransactionID" not in transaction_frame.columns:
        raise ValueError("Bảng transaction thiếu cột TransactionID.")
    if "TransactionID" not in identity_frame.columns:
        raise ValueError("Bảng identity thiếu cột TransactionID.")

    merged_frame = transaction_frame.merge(
        identity_frame,
        on="TransactionID",
        how="left",
        validate="1:1",
    )

    return merged_frame


def create_realtime_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched_frame = frame.copy()

    # {'mục_đích': 'Tạo đặc trưng thời gian để mô hình học được mẫu gian lận theo chu kỳ giờ/ngày/tuần', 'đầu_vào': 'TransactionDT', 'đầu_ra': 'transaction_hour, transaction_day, transaction_week'}
    if "TransactionDT" in enriched_frame.columns:
        enriched_frame["transaction_hour"] = ((enriched_frame["TransactionDT"] // 3600) % 24).astype("float")
        enriched_frame["transaction_day"] = (enriched_frame["TransactionDT"] // (3600 * 24)).astype("float")
        enriched_frame["transaction_week"] = (enriched_frame["TransactionDT"] // (3600 * 24 * 7)).astype("float")

    # {'mục_đích': 'Nén phân phối số tiền để giảm ảnh hưởng outlier khi train', 'đầu_vào': 'TransactionAmt', 'đầu_ra': 'transaction_amt_log1p'}
    if "TransactionAmt" in enriched_frame.columns:
        safe_amount = enriched_frame["TransactionAmt"].fillna(0).clip(lower=0)
        enriched_frame["transaction_amt_log1p"] = np.log1p(safe_amount)

    return enriched_frame


def create_entity_key(frame: pd.DataFrame) -> pd.DataFrame:
    keyed_frame = frame.copy()
    candidate_columns = ["card1", "card2", "addr1", "DeviceInfo"]
    available_columns = [column_name for column_name in candidate_columns if column_name in keyed_frame.columns]

    if not available_columns:
        keyed_frame["entity_key"] = "unknown"
        return keyed_frame

    entity_parts = keyed_frame[available_columns].fillna("na").astype(str)
    keyed_frame["entity_key"] = entity_parts.agg("|".join, axis=1)
    return keyed_frame


def prepare_train_frame(
    data_dir: str,
    chunksize: int = 100_000,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    feature_mode: str = "full",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    if feature_mode not in {"full", "core"}:
        raise ValueError("feature_mode chỉ chấp nhận 'full' hoặc 'core'.")

    transaction_columns = None
    identity_columns = None
    if feature_mode == "core":
        transaction_columns = CORE_TRANSACTION_COLUMNS
        identity_columns = CORE_IDENTITY_COLUMNS

    train_frame = load_ieee_split(
        data_dir=data_dir,
        split="train",
        transaction_columns=transaction_columns,
        identity_columns=identity_columns,
        chunksize=chunksize,
        max_rows=max_rows,
    )
    train_frame = create_realtime_features(train_frame)
    train_frame = create_entity_key(train_frame)

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac phải nằm trong khoảng (0, 1].")
        train_frame = train_frame.sample(frac=sample_frac, random_state=random_state)

    return train_frame


def select_feature_columns(frame: pd.DataFrame, target_column: str = "isFraud") -> List[str]:
    excluded_columns = {target_column, "TransactionID"}
    return [column_name for column_name in frame.columns if column_name not in excluded_columns]


def temporal_train_val_test_split(
    frame: pd.DataFrame,
    target_column: str = "isFraud",
    time_column: str = "TransactionDT",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> TemporalSplit:
    if target_column not in frame.columns:
        raise ValueError(f"Không tìm thấy cột nhãn '{target_column}' trong dữ liệu train.")
    if time_column not in frame.columns:
        raise ValueError(f"Không tìm thấy cột thời gian '{time_column}' để chia dữ liệu theo thời gian.")

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio và val_ratio không hợp lệ.")

    sorted_frame = frame.sort_values(by=time_column).reset_index(drop=True)
    total_rows = len(sorted_frame)

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_frame = sorted_frame.iloc[:train_end]
    val_frame = sorted_frame.iloc[train_end:val_end]
    test_frame = sorted_frame.iloc[val_end:]

    feature_columns = select_feature_columns(sorted_frame, target_column=target_column)

    return TemporalSplit(
        train_x=train_frame[feature_columns],
        train_y=train_frame[target_column].astype(int),
        val_x=val_frame[feature_columns],
        val_y=val_frame[target_column].astype(int),
        test_x=test_frame[feature_columns],
        test_y=test_frame[target_column].astype(int),
    )


def save_parquet_cache(frame: pd.DataFrame, parquet_path: str) -> None:
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
