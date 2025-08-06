import json
import torch
import torchvision.transforms as transforms
import boto3
from io import BytesIO
from PIL import Image
from models import load_custom_resnet
from typing import Generator, List, Optional
from time import perf_counter
import os
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)

resnet_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def fetch_images_from_s3(
    bucket_name: str, num_images: int, cache_dir: Optional[str] = None
) -> Generator[Image.Image, None, None]:
    """
    Fetch up to num_images from the given S3 bucket, ignoring any prefixes or subfolders.
    If cache_dir is provided, save each image into that directory after fetching.
    Yields PIL Images.
    """
    s3 = boto3.client("s3")
    images_fetched = 0
    continuation_token = None

    while images_fetched < num_images:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket_name, ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(Bucket=bucket_name)

        if "Contents" not in response:
            break

        for obj in response["Contents"]:
            if images_fetched >= num_images:
                break

            key = obj["Key"]
            try:
                s3_obj = s3.get_object(Bucket=bucket_name, Key=key)
                img_data = s3_obj["Body"].read()
                pil_image = Image.open(BytesIO(img_data)).convert("RGB")

                # Cache the image locally if a cache directory is provided.
                if cache_dir:
                    file_name = f"{images_fetched}_{os.path.basename(key)}"
                    file_path = os.path.join(cache_dir, file_name)
                    pil_image.save(file_path)

                yield pil_image
                images_fetched += 1
            except Exception as e:
                logger.error(f"Error fetching image {key}: {e}")
                continue

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break


def load_images_from_cache(cache_dir: str, num_images: int) -> List[Image.Image]:
    """
    Load up to num_images from the cache directory.
    """
    cached_image_files = [
        f
        for f in os.listdir(cache_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    cached_image_files.sort()  # sort to ensure consistent order
    images = []
    for file_name in cached_image_files[:num_images]:
        file_path = os.path.join(cache_dir, file_name)
        try:
            img = Image.open(file_path).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.error(f"Error loading cached image {file_name}: {e}")
    return images


def batched(
    iterable: Generator[Image.Image, None, None], n: int
) -> Generator[List[Image.Image], None, None]:
    """
    Group images into batches of size n.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def lambda_handler_resnet(event, context):
    bucket_name = event.get("bucket_name")
    if not bucket_name:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No bucket_name provided in event payload."}),
        }

    num_images = event.get("num_images", 1)  # default to 1 if not specified
    batch_size = event.get("batch_size", 1)  # default to 1 if not specified

    # Define the temporary images directory for caching.
    temp_images_dir = "/tmp/images"
    if not os.path.exists(temp_images_dir):
        os.makedirs(temp_images_dir)

    # Check if there are enough images in the cache folder.
    cached_files = [
        f
        for f in os.listdir(temp_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(cached_files) >= num_images:
        cached_images = load_images_from_cache(temp_images_dir, num_images)
        images_gen = (img for img in cached_images)
    else:
        images_gen = fetch_images_from_s3(
            bucket_name, num_images, cache_dir=temp_images_dir
        )

    # Model loading
    model_name = event.get("model_name", None)
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", None)
        if model_name is None:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "No model_name provided in event payload or env."}
                ),
            }

    s3_models_bucket = os.getenv("S3_BUCKET", "models-experiment-mohammad")
    log_stream_id = context.log_stream_name
    logger.info(
        f"Model: {model_name} - LogStream: {log_stream_id} - Starting execution"
    )

    try:
        model = load_custom_resnet(
            model_name=model_name, s3_bucket=s3_models_bucket, s3_key=f"{model_name}.pth"
        )
    except (ValueError, FileNotFoundError) as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}

    predictions = []
    total_inference_time = 0.0

    for image_batch in batched(images_gen, batch_size):
        # Pre-process batch images.
        batch_tensor = torch.stack(
            [resnet_transform(img) for img in image_batch]  # type: ignore
        )

        start_time = perf_counter()
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predictions += predicted_idx.tolist()
        total_inference_time += perf_counter() - start_time
        
    logger.info(
        f"Total inference time: {total_inference_time:.2f} seconds."
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "model_name": model_name,
                "predicted_labels": predictions,
                "inference_time": total_inference_time,
            }
        ),
    }
