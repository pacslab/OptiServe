import boto3
import json

from botocore.exceptions import (
    ClientError,
    ParamValidationError,
)

from src.exceptions import (
    InvocationError,
    MaxInvocationAttemptsReached,
)

from src.utils.logger import logger
from src.utils.consts import (
    IAM,
    REGION,
)



class StepFunctions:
    
    def __init__(self,
                 state_machine_name: str = None,
                 boto_session: boto3.Session = None,
                 max_invocation_attempts: int = 5,):
        if state_machine_name is None:
            raise ValueError('state_machine_name must be provided')

        self._state_machine_arn = f'arn:aws:states:{REGION}:{IAM}:stateMachine:{state_machine_name}'
        self._state_machine_name = state_machine_name
        self._aws_states_client = boto_session.client('stepfunctions')
        self._max_invocation_attempts = max_invocation_attempts
        
        
    def _invoke(self, payload: str = '{"key1": "value1"}'):
        for _ in range(self._max_invocation_attempts):
            try:
                response = self._aws_states_client.start_execution(
                    stateMachineArn=self._state_machine_arn,
                    input=json.dumps(payload)
                )
                
                return response
            
            except ClientError as e:
                raise InvocationError(message=e.response['Error']['Message'])
            
        raise MaxInvocationAttemptsReached() 