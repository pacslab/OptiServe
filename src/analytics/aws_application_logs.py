import boto3
import time
import datetime
import json

from src.exceptions import (
    MaxInvocationAttemptsReached,
    FunctionTimeout
)

from collections import defaultdict

from src.analytics.aws_logs import AWSLogs


class AWSApplicationLogs(AWSLogs):
    
    def __init__(self,
                 boto_session: boto3.Session = None,
                 application_name: str = None,
                 total_logs_limit: int = 10000):
        super().__init__(boto_session=boto_session)
        
        if application_name is None:
            raise ValueError('application_name must be provided')
        
        self._application_name = application_name
        self._log_group_name = f'/aws/vendedlogs/states/{self._application_name}-Logs'
        self._max_invocation_attempts = 5
        self._total_logs_limit = total_logs_limit
        self._sleep_interval = 1
        
        
    def get_logs(self, start_time: int, end_time: int):
        if start_time is None or end_time is None:
            raise ValueError('start_time and end_time must be provided')
        
        response = self._aws_logs_client.start_query(
            logGroupName=self._log_group_name,
            queryString="fields @timestamp, @message| filter type = 'ExecutionStarted' or type = 'ExecutionSucceeded' | sort id desc",
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
            
            results = defaultdict(lambda: {'s': 0, 'e': 0, 'd': 0})
            for r in response['results']:
                r_json = json.loads(r[1]['value'])
                if r_json['type'] in ['ExecutionSucceeded', 'ExecutionStarted']:
                    execution_arn = r_json['execution_arn'].split(':')[-1]
                    event_timestamp = float(r_json['event_timestamp'])
                    event_type = r_json['type']
                    
                    results[execution_arn]['s' if event_type == 'ExecutionStarted' else 'e'] = event_timestamp

            for key, value in results.items():
                results[key]['d'] = results[key]['e'] - results[key]['s']

            return results


        except MaxInvocationAttemptsReached:
            raise FunctionTimeout("Could not get the logs in time.")