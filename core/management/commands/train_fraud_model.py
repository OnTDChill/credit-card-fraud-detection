import os
import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.models import FraudModelReport
from core.services.fraud.data_pipeline import (
    prepare_train_frame,
    save_parquet_cache,
    temporal_train_val_test_split,
)
from core.services.fraud.training import save_training_artifacts, train_and_evaluate


class Command(BaseCommand):
    help = "Huấn luyện mô hình phát hiện fraud từ bộ IEEE-CIS theo split thời gian."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=getattr(settings, "FRAUD_IEEE_DATA_DIR", ""),
            help="Đường dẫn thư mục chứa train/test CSV của IEEE-CIS.",
        )
        parser.add_argument(
            "--artifacts-dir",
            type=str,
            default=getattr(settings, "FRAUD_ARTIFACTS_DIR", ""),
            help="Đường dẫn lưu model và báo cáo metrics.",
        )
        parser.add_argument(
            "--chunksize",
            type=int,
            default=getattr(settings, "FRAUD_TRAIN_CHUNK_SIZE", 100000),
            help="Kích thước chunk khi đọc file transaction lớn.",
        )
        parser.add_argument(
            "--sample-frac",
            type=float,
            default=None,
            help="Lấy mẫu dữ liệu train theo tỷ lệ để chạy thử nhanh (0-1].",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=getattr(settings, "FRAUD_MODEL_RANDOM_STATE", 42),
            help="Seed để tái lập kết quả train.",
        )
        parser.add_argument(
            "--skip-parquet-cache",
            action="store_true",
            help="Bỏ qua bước lưu parquet cache trung gian.",
        )
        parser.add_argument(
            "--feature-mode",
            type=str,
            default="core",
            choices=["core", "full"],
            help="Chế độ cột dữ liệu: core để train nhanh, full để benchmark đầy đủ.",
        )
        parser.add_argument(
            "--max-rows",
            type=int,
            default=5000,
            help="Giới hạn số dòng đọc từ IEEE để smoke test nhanh.",
        )
        parser.add_argument(
            "--latency-budget-ms",
            type=float,
            default=getattr(settings, "FRAUD_LATENCY_BUDGET_MS", 120.0),
            help="Ngưỡng p95 latency mục tiêu (ms) để chọn champion model.",
        )

    def handle(self, *args, **options):
        project_root = settings.BASE_DIR

        data_dir = options["data_dir"] or os.path.join(project_root, "data", "ieee-fraud-detection")
        artifacts_dir = options["artifacts_dir"] or os.path.join(project_root, "artifacts", "fraud")
        chunksize = options["chunksize"]
        sample_frac = options["sample_frac"]
        random_state = options["random_state"]
        skip_parquet_cache = options["skip_parquet_cache"]
        feature_mode = options["feature_mode"]
        max_rows = options["max_rows"]
        latency_budget_ms = options["latency_budget_ms"]

        if not os.path.exists(data_dir):
            raise CommandError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

        self.stdout.write(self.style.NOTICE("Đang nạp và chuẩn hóa dữ liệu IEEE-CIS..."))
        train_frame = prepare_train_frame(
            data_dir=data_dir,
            chunksize=chunksize,
            sample_frac=sample_frac,
            random_state=random_state,
            feature_mode=feature_mode,
            max_rows=max_rows,
        )

        if "isFraud" not in train_frame.columns:
            raise CommandError("Dữ liệu train không chứa cột nhãn isFraud.")

        if not skip_parquet_cache:
            parquet_path = os.path.join(artifacts_dir, "cache", "train_master.parquet")
            save_parquet_cache(train_frame, parquet_path)
            self.stdout.write(self.style.SUCCESS(f"Đã lưu parquet cache: {parquet_path}"))

        self.stdout.write(self.style.NOTICE("Đang chia train/validation/test theo thời gian..."))
        temporal_split = temporal_train_val_test_split(train_frame)

        self.stdout.write(self.style.NOTICE("Đang huấn luyện và đánh giá mô hình fraud..."))
        training_result = train_and_evaluate(
            train_x=temporal_split.train_x,
            train_y=temporal_split.train_y,
            val_x=temporal_split.val_x,
            val_y=temporal_split.val_y,
            test_x=temporal_split.test_x,
            test_y=temporal_split.test_y,
            random_state=random_state,
            latency_budget_ms=latency_budget_ms,
        )

        artifact_paths = save_training_artifacts(
            pipeline=training_result["pipeline"],
            report=training_result["report"],
            artifacts_dir=artifacts_dir,
            prediction_frames=training_result.get("prediction_frames"),
        )

        self.stdout.write(self.style.SUCCESS("Huấn luyện hoàn tất."))
        self.stdout.write(self.style.SUCCESS(f"Model: {artifact_paths['model_path']}"))
        self.stdout.write(self.style.SUCCESS(f"Report: {artifact_paths['report_path']}"))
        for split_name in ["train", "validation", "test"]:
            prediction_key = f"predictions_{split_name}_path"
            if prediction_key in artifact_paths:
                self.stdout.write(
                    self.style.SUCCESS(f"Predictions ({split_name}): {artifact_paths[prediction_key]}")
                )

        try:
            FraudModelReport.objects.filter(is_champion=True).update(is_champion=False)
            FraudModelReport.objects.create(
                model_name=training_result["report"].get("model", "unknown"),
                model_version=training_result["report"].get("model_version", "unknown"),
                is_champion=True,
                report_json=json.dumps(training_result["report"], ensure_ascii=False),
            )
            self.stdout.write(self.style.SUCCESS("Đã cập nhật FraudModelReport champion trong database."))
        except Exception as error:  # noqa: BLE001
            self.stdout.write(
                self.style.WARNING(
                    f"Không thể lưu FraudModelReport vào DB (có thể chưa migrate): {error}"
                )
            )
