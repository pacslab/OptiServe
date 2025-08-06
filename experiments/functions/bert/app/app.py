import json
import torch
import boto3
import logging
from time import perf_counter
from models import load_custom_bert
import os
import transformers

logger = logging.getLogger()
logger.setLevel(logging.INFO)


TOKENIZED_CACHE_PATH = "/tmp/texts.json"


def load_tokenized_inputs_or_tokenize(event, tokenizer):
    if os.path.exists(TOKENIZED_CACHE_PATH):
        tensor_dict = torch.load(TOKENIZED_CACHE_PATH)
        tokenized_inputs = transformers.BatchEncoding(tensor_dict)
        logger.info("Loaded tokenized inputs from cache.")
        return tokenized_inputs
    else:
        sample_text = (
            "The field of machine learning has seen tremendous growth over the past decade. "
            "Deep learning, in particular, has revolutionized areas such as image recognition, "
            "speech processing, and natural language understanding. "
            "Large-scale pre-trained models like BERT and GPT have set new performance records across many benchmarks. "
            "However, the computational demands of these models pose challenges for real-time inference and deployment. "
        )
        texts = [sample_text * 8 for _ in range(16)]
        tokenized_inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        torch.save(tokenized_inputs.data, TOKENIZED_CACHE_PATH)
        logger.info("Tokenized synthetic inputs cached to /tmp.")
        return tokenized_inputs


def lambda_handler_bert(event, context):        
    model_name = event.get("model_name")
    if not model_name:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No model_name provided."}),
        }

    log_stream_id = context.log_stream_name
    logger.info(f"Model: {model_name} - LogStream: {log_stream_id} - Starting execution")

    s3_models_bucket = os.getenv("S3_BUCKET", "models-experiment-mohammad")

    try:
        model, tokenizer = load_custom_bert(
            model_name=model_name,
            s3_bucket=s3_models_bucket,
            s3_key=f"{model_name}/pytorch_model.bin"
        )
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Failed to load model {model_name}"})
        }

    tokenized_inputs = load_tokenized_inputs_or_tokenize(event, tokenizer)
    if tokenized_inputs is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No texts provided."})
        }

    start_time = perf_counter()
    with torch.no_grad():
        _ = model(**tokenized_inputs)
    total_inference_time = perf_counter() - start_time
    
    logger.info(f"Inference completed in {total_inference_time:.4f} seconds.")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "model_name": model_name,
            "inference_time": total_inference_time
        }),
    }