import os
import boto3
import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from datetime import datetime, timedelta
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
logs_client = boto3.client('logs')
sagemaker_client = boto3.client('sagemaker')
step_functions_client = boto3.client('stepfunctions')


def create_applications(applications: List[Application]=[]):
    pass


def create_functions(application_dir: str = None): 
    functions: List[Function] = []
    
    for function_dir in os.listdir(application_dir):
        if (not os.path.isdir(os.path.join(application_dir, function_dir))) or function_dir == '__pycache__':
            continue
        function = Function(code_dir=os.path.join(application_dir, function_dir), name=f'{application_dir.split('/')[-1]}_{function_dir}')
        functions.append(function)
        
        
    for function in functions:
        try:
            response = logs_client.create_log_group(logGroupName=f'/aws/lambda/{function.name}')
            logger.info(f"Log Group for function {function.name} created successfully.")
        except Exception as e:
            logger.error(f"Failed to create Log Group for function {function.name}: {e}")


    for function in functions:
        zip_dir(function.code_dir, f'{function.name}.zip')
        zip_file = get_zip_file_as_bytes(f'{function.name}.zip')
        
        log_group_name = f'/aws/lambda/{function.name}'
        
        try:
            response = lambda_client.create_function(
            FunctionName=function.name,
            Runtime=function.runtime,
            Role=f'arn:aws:iam::{IAM}:role/{LAMBDA_ROLE}',
            Handler='lambda_function.lambda_handler',
            Code={
                'ZipFile': zip_file
            },
            Description='Function created by the Function as a Service (FaaS) platform',
            MemorySize=function.memory_size,
            Timeout=60,
            Publish=True,
            Environment={
                    'Variables': {
                        'LOG_GROUP_NAME': log_group_name
                    }
                }
            )
            
            logger.info(f"Function {function.name} created successfully.")
            print(f"Function {function.name} created successfully.")
            
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
        
        delete_file(f'{function.name}.zip')
        time.sleep(1)
        
    return functions


def profile_application(application: Application=None, num_of_iterations=100):
    pass


def profile_function(function: Function=None, num_of_iterations=100, need_update=False):        
    if need_update:
        lambda_client.update_function_configuration(FunctionName=function.name, MemorySize=function.memory_size)
        logger.info(f"Function {function.name} memory size updated to {function.memory_size} MB.")
        time.sleep(1)
        
    
    for _ in tqdm.tqdm(range(num_of_iterations)):
        time.sleep(10)
        res = lambda_client.invoke(FunctionName=function.name, InvocationType='Event')
        
    logger.info(f"Function {function.name} invoked {num_of_iterations} times.")
        
        
def get_function_profiling_logs(log_group_name=None):
    if log_group_name is None:
        raise ValueError("log_group_name cannot be None.")

    start_time = int((datetime.utcnow() - timedelta(days=30)).timestamp())  # Last month
    end_time = int(datetime.utcnow().timestamp())
    
    
    results: List[PerformanceMetrics] = []
    
        
    response = logs_client.start_query(
        logGroupName=log_group_name,
        queryString="fields @timestamp, @message| filter @message like 'REPORT'| sort @timestamp desc",
        startTime=start_time,
        endTime=end_time,
        limit=10000
    )
    
    query_id = response['queryId']
    response = None
    
    while response == None or response['status'] == 'Running':
        print('Waiting for query to complete ...')
        time.sleep(1)
        response = logs_client.get_query_results(
            queryId=query_id
        )

    for r in response['results']:
        timestamp = r[0]['value']
        log_list = [item.split(': ') for item in r[1]['value'].split('\t')][:-1]
        
        results.append(PerformanceMetrics(
            invocation_time=timestamp,
            max_memory_usage=int(log_list[4][1].split(' ')[0]),
            memory_usage=int(log_list[3][1].split(' ')[0]),
            billable_duration=int(log_list[2][1].split(' ')[0]),
            duration=float(log_list[1][1].split(' ')[0]),
            init_duration=float(log_list[5][1].split(' ')[0]) if len(log_list) == 6 else None,
        ))

    return results