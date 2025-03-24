import base64
import boto3
import time

from botocore.exceptions import (
    ClientError,
    ReadTimeoutError,
    ParamValidationError,
)

from src.utils.logger import logger
from src.exceptions import (
    InvocationError,
    FunctionTimeout,
    MaxInvocationAttemptsReached,
)


class Invoker:
    def __init__(self, function_name: str, max_invocations: int, boto_session: boto3.Session):
        self._function_name = function_name
        self._max_invocations = max_invocations
        self._aws_lambda_client = boto3.client('lambda')
        
    
    def _invoke(self, payload: str):
        memory_mb = None
        timeout_s = None
        sleeping_interval = 1
        
        for _ in range(self._max_invocations):
            try:
                config = self._aws_lambda_client.get_function_configuration(FunctionName=self._function_name)
                
                memory_mb = config['MemorySize']
                timeout_s = config['Timeout']
                
                logger.info(f'Trying to invoke {self._function_name} with memory {memory_mb} MB and timeout {timeout_s} s with payload {payload}')
                
                response = self._aws_lambda_client.invoke(
                    FunctionName=self._function_name,
                    LogType='Tail',
                    Payload=payload,
                )
                
                return response
                
            except ClientError as e:
                if e.response["Error"]["Code"] == "TooManyRequestsException":
                    logger.debug(f'Concurrent Invocation Limit Exceeded. Retrying... {self._function_name}: {memory_mb}MB')

                    time.sleep(sleeping_interval)
                    sleeping_interval *= 2

                else:
                    logger.debug(e.args[0])
                    raise InvocationError(
                        "Error has been raised while invoking the lambda function. Please make sure "
                        "that the provided function name and configuration are correct!"
                    )
                    
            except ReadTimeoutError:
                logger.warning(f"Lambda exploration timed out. {self._function_name}: {memory_mb}MB")
                raise FunctionTimeout(duration_ms=timeout_s)


            except ParamValidationError as e:
                raise InvocationError(e.args[0])
            
            
            except Exception as e:
                time.sleep(sleeping_interval)
                sleeping_interval *= 2
        
        logger.warning(f"Max invocation attempts reached. {self._function_name}: {memory_mb}MB")   
        raise MaxInvocationAttemptsReached()
    
    
    
    def invoke_to_get_duration(self, payload: str):
        response = self._invoke(payload)
        
        response = str(base64.b64decode(response['LogResult']))
        
        return response
    
    
    def invoke_with_payload(self, payload: str):
        response = self._invoke(payload)
        
        response_payload = response['Payload'].read().decode('utf-8')
        
        return response_payload