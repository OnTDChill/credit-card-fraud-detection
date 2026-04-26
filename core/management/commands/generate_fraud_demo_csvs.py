import os
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.services.fraud.data_pipeline import prepare_train_frame


def _parse_ratios(raw_value: str) -> List[float]:
    ratios = []
    for token in raw_value.split(","):
        value = float(token.strip())
        if value <= 0 or value >= 1:
            raise ValueError("Mỗi fraud ratio phải nằm trong khoảng (0, 1).")
        ratios.append(value)
    if not ratios:
        raise ValueError("Cần cung cấp ít nhất một fraud ratio.")
    return ratios


def _sample_class(pool: pd.DataFrame, target_rows: int, random_state: int) -> pd.DataFrame:
    if target_rows <= 0:
        return pool.iloc[:0].copy()
    replace_mode = target_rows > len(pool)
    return pool.sample(n=target_rows, replace=replace_mode, random_state=random_state)


def _to_upload_schema(sampled_frame: pd.DataFrame, batch_prefix: str) -> pd.DataFrame:
    utc_now = datetime.now(timezone.utc)
    base_timestamp = int(utc_now.timestamp())

    upload_frame = sampled_frame.reset_index(drop=True).copy()

    upload_frame["external_transaction_id"] = [
        f"{batch_prefix}-txn-{index + 1:08d}" for index in range(len(upload_frame))
    ]

    user_seed = upload_frame.get("card1", pd.Series(np.arange(len(upload_frame)))).fillna(0).astype(int)
    card_seed = upload_frame.get("card2", pd.Series(np.arange(len(upload_frame)))).fillna(0).astype(int)

    upload_frame["external_user_id"] = [f"demo-user-{value % 100000:05d}" for value in user_seed]
    upload_frame["card_fingerprint"] = [f"demo-card-{value % 100000:05d}" for value in card_seed]

    if "TransactionAmt" not in upload_frame.columns:
        upload_frame["TransactionAmt"] = np.random.default_rng(42).uniform(5.0, 500.0, size=len(upload_frame))

    upload_frame["amount"] = upload_frame["TransactionAmt"].fillna(0).astype(float)
    upload_frame["currency"] = "usd"
    upload_frame["ip_address"] = [f"10.10.{(index % 200) + 1}.{(index % 240) + 10}" for index in range(len(upload_frame))]

    if "DeviceInfo" in upload_frame.columns:
        upload_frame["device_info"] = upload_frame["DeviceInfo"].fillna("streamlit-device")
    else:
        upload_frame["device_info"] = "streamlit-device"

    if "TransactionDT" in upload_frame.columns:
        upload_frame["transaction_dt"] = upload_frame["TransactionDT"].fillna(base_timestamp).astype(int)
    else:
        upload_frame["transaction_dt"] = [base_timestamp + index for index in range(len(upload_frame))]

    upload_frame["event_time"] = [
        datetime.fromtimestamp(base_timestamp + index, tz=timezone.utc).isoformat()
        for index in range(len(upload_frame))
    ]

    upload_frame["is_fraud_truth"] = upload_frame["isFraud"].astype(int)

    ordered_columns = [
        "external_transaction_id",
        "external_user_id",
        "card_fingerprint",
        "amount",
        "currency",
        "ip_address",
        "device_info",
        "event_time",
        "transaction_dt",
        "is_fraud_truth",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "card2",
        "addr1",
        "C1",
        "D1",
        "DeviceInfo",
    ]
    available_columns = [column_name for column_name in ordered_columns if column_name in upload_frame.columns]
    return upload_frame[available_columns]


def _write_large_batch_csv(
    output_file: str,
    fraud_pool: pd.DataFrame,
    normal_pool: pd.DataFrame,
    fraud_ratio: float,
    total_rows: int,
    random_state: int,
    batch_prefix: str,
    chunk_size: int = 50_000,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    first_chunk = True
    generated_rows = 0
    chunk_index = 0

    while generated_rows < total_rows:
        current_chunk_size = min(chunk_size, total_rows - generated_rows)
        fraud_rows = int(round(current_chunk_size * fraud_ratio))
        normal_rows = current_chunk_size - fraud_rows

        sampled_fraud = _sample_class(fraud_pool, fraud_rows, random_state + chunk_index * 3 + 1)
        sampled_normal = _sample_class(normal_pool, normal_rows, random_state + chunk_index * 3 + 2)
        sampled_frame = pd.concat([sampled_fraud, sampled_normal], ignore_index=True)
        sampled_frame = sampled_frame.sample(frac=1.0, random_state=random_state + chunk_index * 3 + 3).reset_index(drop=True)

        chunk_prefix = f"{batch_prefix}-chunk-{chunk_index + 1:04d}"
        upload_frame = _to_upload_schema(sampled_frame, batch_prefix=chunk_prefix)

        upload_frame.to_csv(output_file, mode="w" if first_chunk else "a", header=first_chunk, index=False)

        generated_rows += len(upload_frame)
        chunk_index += 1
        first_chunk = False


class Command(BaseCommand):
    help = "Tạo các file CSV demo để upload batch fraud lên realtime monitor."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=getattr(settings, "FRAUD_IEEE_DATA_DIR", ""),
            help="Thư mục chứa dữ liệu IEEE-CIS.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="",
            help="Thư mục ghi các file CSV demo.",
        )
        parser.add_argument(
            "--rows",
            type=int,
            default=50_000,
            help="Số giao dịch mỗi file demo ratio-based.",
        )
        parser.add_argument(
            "--ratios",
            type=str,
            default="0.20,0.25,0.30",
            help="Danh sách fraud ratio, cách nhau bởi dấu phẩy.",
        )
        parser.add_argument(
            "--include-realistic-ratio",
            action="store_true",
            help="Sinh thêm một file với tỷ lệ fraud gần thực tế từ train set.",
        )
        parser.add_argument(
            "--include-test-scale",
            action="store_true",
            help="Sinh thêm một file có số lượng dòng bằng test_transaction.csv mặc định.",
        )
        parser.add_argument(
            "--test-scale-rows",
            type=int,
            default=506_691,
            help="Số dòng cho file test-scale.",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=42,
            help="Seed random để tái lập kết quả.",
        )

    def handle(self, *args, **options):
        project_root = settings.BASE_DIR

        data_dir = options["data_dir"] or os.path.join(project_root, "data", "ieee-fraud-detection")
        output_dir = options["output_dir"] or os.path.join(project_root, "data", "fraud-demo-batches")
        rows = int(options["rows"])
        random_state = int(options["random_state"])

        if rows <= 0:
            raise CommandError("--rows phải lớn hơn 0.")
        if not os.path.exists(data_dir):
            raise CommandError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

        try:
            ratios = _parse_ratios(options["ratios"])
        except ValueError as error:
            raise CommandError(str(error)) from error

        os.makedirs(output_dir, exist_ok=True)

        self.stdout.write(self.style.NOTICE("Đang chuẩn bị dữ liệu train để tạo file batch demo..."))
        train_frame = prepare_train_frame(
            data_dir=data_dir,
            feature_mode="core",
            max_rows=None,
            random_state=random_state,
        )

        if "isFraud" not in train_frame.columns:
            raise CommandError("Không tìm thấy cột isFraud trong train set.")

        fraud_pool = train_frame[train_frame["isFraud"] == 1]
        normal_pool = train_frame[train_frame["isFraud"] == 0]

        if fraud_pool.empty or normal_pool.empty:
            raise CommandError("Không đủ dữ liệu để tạo batch fraud/normal.")

        generated_files: List[str] = []

        for ratio in ratios:
            fraud_rows = int(round(rows * ratio))
            normal_rows = rows - fraud_rows

            sampled_fraud = _sample_class(fraud_pool, fraud_rows, random_state)
            sampled_normal = _sample_class(normal_pool, normal_rows, random_state + 7)
            sampled_frame = pd.concat([sampled_fraud, sampled_normal], ignore_index=True)
            sampled_frame = sampled_frame.sample(frac=1.0, random_state=random_state + 11).reset_index(drop=True)

            batch_prefix = f"fraud-ratio-{int(ratio * 100):02d}"
            upload_frame = _to_upload_schema(sampled_frame, batch_prefix=batch_prefix)

            output_file = os.path.join(output_dir, f"batch_{batch_prefix}_rows_{rows}.csv")
            upload_frame.to_csv(output_file, index=False)
            generated_files.append(output_file)

        if options.get("include_realistic_ratio"):
            realistic_ratio = float(train_frame["isFraud"].mean())
            fraud_rows = int(round(rows * realistic_ratio))
            normal_rows = rows - fraud_rows

            sampled_fraud = _sample_class(fraud_pool, fraud_rows, random_state + 19)
            sampled_normal = _sample_class(normal_pool, normal_rows, random_state + 23)
            sampled_frame = pd.concat([sampled_fraud, sampled_normal], ignore_index=True)
            sampled_frame = sampled_frame.sample(frac=1.0, random_state=random_state + 29).reset_index(drop=True)

            upload_frame = _to_upload_schema(sampled_frame, batch_prefix="fraud-realistic")
            output_file = os.path.join(
                output_dir,
                f"batch_fraud_realistic_{realistic_ratio:.4f}_rows_{rows}.csv",
            )
            upload_frame.to_csv(output_file, index=False)
            generated_files.append(output_file)

        if options.get("include_test_scale"):
            large_rows = int(options["test_scale_rows"])
            if large_rows <= 0:
                raise CommandError("--test-scale-rows phải lớn hơn 0.")

            ratio_for_large = ratios[0]
            output_file = os.path.join(
                output_dir,
                f"batch_fraud_test_scale_ratio_{ratio_for_large:.2f}_rows_{large_rows}.csv",
            )

            _write_large_batch_csv(
                output_file=output_file,
                fraud_pool=fraud_pool,
                normal_pool=normal_pool,
                fraud_ratio=ratio_for_large,
                total_rows=large_rows,
                random_state=random_state + 31,
                batch_prefix="fraud-test-scale",
            )

            generated_files.append(output_file)

        self.stdout.write(self.style.SUCCESS("Đã tạo xong các file CSV demo:"))
        for file_path in generated_files:
            self.stdout.write(self.style.SUCCESS(f"- {file_path}"))
