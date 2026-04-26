import os
from functools import lru_cache
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from core.services.fraud.data_pipeline import create_realtime_features


def _default_model_path() -> str:
    artifacts_dir = getattr(
        settings,
        'FRAUD_ARTIFACTS_DIR',
        os.path.join(settings.BASE_DIR, 'artifacts', 'fraud')
    )
    return os.path.join(artifacts_dir, 'fraud_model.joblib')


@lru_cache(maxsize=1)
def _load_cached_pipeline(model_path: str, model_mtime: float):
    del model_mtime
    return joblib.load(model_path)


def load_champion_pipeline() -> Tuple[object, str, str]:
    model_path = _default_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Không tìm thấy model đã train: {model_path}')

    model_mtime = os.path.getmtime(model_path)
    pipeline = _load_cached_pipeline(model_path, model_mtime)
    model_name = pipeline.named_steps['model'].__class__.__name__
    model_version = str(int(model_mtime))
    return pipeline, model_name, model_version


def build_inference_frame(pipeline, payload_features: Dict[str, object]) -> pd.DataFrame:
    preprocessor = pipeline.named_steps['preprocessor']
    expected_columns = list(preprocessor.feature_names_in_)

    default_row = {column_name: np.nan for column_name in expected_columns}
    for key, value in payload_features.items():
        if key in default_row:
            default_row[key] = value

    frame = pd.DataFrame([default_row])
    frame = create_realtime_features(frame)

    # {'mục_đích': 'Bảo đảm DataFrame suy luận luôn đủ mọi cột mà pipeline đã fit', 'đầu_vào': 'payload feature rời rạc từ API', 'đầu_ra': 'one-row inference frame với cột thiếu được điền NaN'}
    for column_name in expected_columns:
        if column_name not in frame.columns:
            frame[column_name] = np.nan

    return frame[expected_columns]


def predict_fraud_score(pipeline, payload_features: Dict[str, object]) -> float:
    inference_frame = build_inference_frame(pipeline, payload_features)
    probabilities = pipeline.predict_proba(inference_frame)[:, 1]
    return float(probabilities[0])
