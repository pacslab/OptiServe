import os
import boto3
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cache of loaded models to avoid re-initialization on each invocation
BERT_MODELS = {}


def load_custom_bert(model_name: str, s3_bucket: str, s3_key: str):
    """
    Loads a bert model.
    Caches it in memory so subsequent invocations can reuse it (if Lambda stays warm).
    """
    key = f"{model_name}-{s3_key}"

    if key not in BERT_MODELS:
        tmp_model_path = f"/tmp/{model_name}"

        if not os.path.exists(tmp_model_path):
            os.makedirs(tmp_model_path, exist_ok=True)
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            with open(f"{tmp_model_path}/pytorch_model.bin", "wb") as f:
                f.write(response["Body"].read())
            logger.info(f"Model {model_name} downloaded to {tmp_model_path}")

        model = AutoModelForSequenceClassification.from_pretrained(tmp_model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        logger.info(f"Model {model_name} loaded successfully.")

        model.eval()
        BERT_MODELS[key] = (model, tokenizer)

    return BERT_MODELS[key]
