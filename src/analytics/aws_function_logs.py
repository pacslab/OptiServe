import boto3
import time
import datetime


import numpy as np
import pandas as pd


from .aws_logs import AWSLogs

from src.exceptions import (
    FunctionTimeout,
    MaxInvocationAttemptsReached,
)
from src.utils.logger import logger


class AWSFunctionLogs(AWSLogs):
    def __init__(self,
                 boto_session: boto3.Session = None,
                 function_name: str = None,
                 total_logs_limit: int = 10000):
        super().__init__(boto_session=boto_session)
        if function_name is None:
            raise ValueError('function_name must be provided')
        
        self._function_name = function_name
        self._log_group_name = f'/aws/lambda/{self._function_name}'
        self._total_logs_limit = total_logs_limit
        self._max_invocation_attempts = 5
        self._sleep_interval = 1
        
        
    def get_logs(self, start_time: int, end_time: int):
        if start_time is None or end_time is None:
            raise ValueError('start_time and end_time must be provided')
        
        response = self._aws_logs_client.start_query(
            logGroupName=self._log_group_name,
            queryString="fields @timestamp, @message| filter @message like 'REPORT'| sort @timestamp desc",
            startTime=start_time,
            endTime=end_time,
            limit=self._total_logs_limit
        )
        
        query_id = response['queryId']
        response = None
        
        
        try:
            attempts = 0
            while attempts < self._max_invocation_attempts:
                response = self._aws_logs_client.get_query_results(
                    queryId=query_id
                )
                
                if response['status'] == 'Complete':
                    break
                
                time.sleep(self._sleep_interval)
                attempts += 1
                
            if response['status'] != 'Complete':
                raise MaxInvocationAttemptsReached()
            
            results = []
            for r in response['results']:
                parsed_log = self.log_parser.parse_function_profiling_logs(r[1]['value'])
                parsed_log['Timestamp'] = r[0]['value']
                
                results.append(parsed_log)
                
            return results


        except MaxInvocationAttemptsReached:
            raise FunctionTimeout("Could not get the logs in time.")
        
        
    def get_logs_df(self, start_time: int, end_time: int):
        if start_time is None or end_time is None:
            raise ValueError('start_time and end_time must be provided')

        logs = self.get_logs(start_time=start_time, end_time=end_time)
        
        logs = pd.DataFrame(logs)
        
        return logs