import boto3
import torch
import torchvision.models as models
import os

# Cache of loaded models to avoid re-initialization on each invocation
RESNET_MODELS = {}


def load_custom_resnet(model_name: str, s3_bucket: str, s3_key: str):
    """
    Loads a ResNet model from an S3 bucket.
    Caches it in memory so subsequent invocations can reuse it (if Lambda stays warm).

    Parameters:
        model_name: Name of the ResNet model variant.
        s3_bucket: Name of the S3 bucket where the model file is stored.
        s3_key: S3 key (path) to the model file.
    """
    if not s3_bucket or not s3_key:
        raise ValueError("Both s3_bucket and s3_key must be provided.")

    key = f"{model_name}-{s3_key}"

    if key not in RESNET_MODELS:
        if model_name == "resnet-18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet-34":
            model = models.resnet34(weights=None)
        elif model_name == "resnet-50":
            model = models.resnet50(weights=None)
        elif model_name == "resnet-101":
            model = models.resnet101(weights=None)
        elif model_name == "resnet-152":
            model = models.resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        tmp_file = f"/tmp/{model_name}.pth"

        if not os.path.exists(tmp_file):
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            data = response["Body"].read()

            with open(tmp_file, "wb") as f:
                f.write(data)

        state_dict = torch.load(tmp_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        RESNET_MODELS[key] = model

    return RESNET_MODELS[key]
