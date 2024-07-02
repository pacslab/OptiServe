import os
import boto3
import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from botocore.exceptions import ClientError

from ..utils.Application import Application
from ..utils.Function import Function
from ..utils.Logger import Logger
from ..utils.PerformanceMetrics import PerformanceMetrics
from ..utils.consts import (
    MEM_OPTS,
    IAM,
    LAMBDA_ROLE
)
from ..utils.zipper import (
    zip_dir,
    unzip_dir,
    get_zip_file_as_bytes,
    delete_file
)


# This script is responsible for profiling the applications and functions on AWS Lambda using the boto3 library.

logger = Logger()

lambda_client = boto3.client('lambda')
sagemaker_client = boto3.client('sagemaker')
step_functions_client = boto3.client('stepfunctions')


def create_applications(step_functions_client=None, applications: List[Application]=[]):
    if not step_functions_client:
        raise ValueError("Please provide a valid step functions client")


def create_functions(lambda_client=None, application_dir: str = None):
    if not lambda_client:
        lambda_client = boto3.client('lambda')
    
    functions: List[Function] = []
    
    for function_dir in os.listdir(application_dir):
        function = Function(code_dir=os.path.join(application_dir, function_dir), name=f'{application_dir.split('/')[-1]}_{function_dir}')
        functions.append(function)

    for function in functions:
        zip_dir(function.code_dir, f'{function.name}.zip')
        zip_file = get_zip_file_as_bytes(f'{function.name}.zip')
        
        try:
            response = lambda_client.create_function(
            FunctionName=function.name,
            Runtime=function.runtime,
            Role=f'arn:aws:iam::{IAM}:role/{LAMBDA_ROLE}',
            Handler='function.lambda_handler',
            Code={
                'ZipFile': zip_file
            },
            Description='Function created by the Function as a Service (FaaS) platform',
            MemorySize=function.memory_size,
            Timeout=60,
            Publish=True
            )
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                logger.warning(f"Function {function.name} already exists.")
                try:
                    response = lambda_client.update_function_code(
                        FunctionName=function.name,
                        ZipFile=zip_file,
                        Publish=True
                    )
                    logger.info(f"Function {function.name} code updated successfully.")
                except ClientError as update_error:
                    logger.error(f"Failed to update function code: {update_error}")
            else:
                logger.error(f"Unexpected error: {e}")
        
        logger.info(f"Function {function.name} created successfully with ARN: {response['FunctionArn']}")
        print(f"Function {function.name} created successfully.")
        delete_file(f'{function.name}.zip')
        time.sleep(1)
        
    return functions


def profile_application(step_functions_client=None, application: Application=None, num_of_iterations=100):
    if not step_functions_client:
        raise ValueError("Please provide a valid step functions client")


def profile_function(lambda_client=None, function: Function=None, num_of_iterations=100):
    if not lambda_client:
        lambda_client = boto3.client('lambda')
        
    lambda_client.update_function_configuration(FunctionName=function.name, MemorySize=function.memory_size)
    time.sleep(1)
    
    for _ in tqdm.tqdm(range(num_of_iterations)):
        time.sleep(10)
        res = lambda_client.invoke(FunctionName=function.name, InvocationType='Event')
        print(res)
    