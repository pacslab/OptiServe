import os
import boto3
from ultralytics import YOLO

# Cache of loaded models to avoid re-initialization on each invocation
YOLO_MODELS = {}


def load_custom_yolo(model_name: str, s3_bucket: str, s3_key: str) -> YOLO:
    """
    Loads a yolo model from a local .pt file.
    Caches it in memory so subsequent invocations can reuse it (if Lambda stays warm).
    """
    key = f"{model_name}-{s3_key}"

    if key not in YOLO_MODELS:
        tmp_file = f"/tmp/{model_name}.pt"

        if not os.path.exists(tmp_file):
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            data = response["Body"].read()

            with open(tmp_file, "wb") as f:
                f.write(data)

        model = YOLO(tmp_file)

        YOLO_MODELS[key] = model

    return YOLO_MODELS[key]
