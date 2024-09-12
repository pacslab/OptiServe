import boto3
from botocore.exceptions import (
    ClientError,
    ParamValidationError,
)

from .config import FunctionConfig

from src.utils.logger import logger
from src.exceptions import (
    FunctionConfigurationError,
)


class ConfigManager:
    
    def __init__(self, function_name: str, boto_session: boto3.Session):
        self._function_name = function_name
        self._initial_config = None
        self._aws_lambda_client = boto_session.client('lambda')
        self._aws_quotas_client = boto_session.client('service-quotas')
        
        
    def _get_max_timeout(self):
        try:
            quota = self._aws_quotas_client.get_service_quota(
                ServiceCode="lambda", QuotaCode="L-9FEEFFC0"
            )

        except ClientError:
            quota = self._aws_quotas_client.get_aws_default_service_quota(
                ServiceCode="lambda", QuotaCode="L-9FEEFFC0"
            )

        return int(quota["Quota"]["Value"])
    
    
    def set_config(self, memory_mb: int, timeout_s: int = None):
        try:
            config = self._aws_lambda_client.get_function_configuration(FunctionName=self._function_name)
            
            if not self._initial_config:
                self._initial_config = FunctionConfig(config['MemorySize'], config['Timeout'])
                
            if timeout_s:
                self._aws_lambda_client.update_function_configuration(
                    FunctionName=self._function_name,
                    MemorySize=int(memory_mb),
                    Timeout=timeout_s,
                )
            else:
                self._aws_lambda_client.update_function_configuration(
                    FunctionName=self._function_name,
                    MemorySize=int(memory_mb),
                    Timeout=self._get_max_timeout(),
                )
                
            while (
                config['MemorySize'] != memory_mb or
                config['LastUpdateStatus'] == 'InProgress'
            ):
                w = self._aws_lambda_client.get_waiter('function_updated')
                w.wait(FunctionName=self._function_name)
                
                config = self._aws_lambda_client.get_function_configuration(FunctionName=self._function_name)
        
        except ParamValidationError as e:
            logger.debug(e.args[0])
            raise FunctionConfig(e.args[0])
        

        except ClientError as e:

            if e.response['Error']['Code'] == 'ResourceConflictException':
                logger.warning("Concurrent Update Function Error. Retrying ...")

                time.sleep(2)

                self.set_config(memory_mb, timeout_s)

            else:
                raise FunctionConfigurationError(e.args[0])

        else:
            return config
    
    
    def reset_config(self):
        self.set_config(self._initial_config.memory_mb, self._initial_config.timeout_s)