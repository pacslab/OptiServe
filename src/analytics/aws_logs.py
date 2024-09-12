from abc import ABC, abstractmethod

import boto3


class AWSLogs(ABC):
    def __init__(self,
                 boto_session: boto3.Session = None):
        self._aws_logs_client = boto_session.client('logs')
        
    @abstractmethod
    def get_logs(self, *args, **kwargs):
        pass