import boto3
import time
from botocore.exceptions import (
    ClientError,
    ParamValidationError,
)

from src.profiler.function_config import FunctionConfig

from src.utils.logger import logger
from src.exceptions import (
    FunctionConfigurationError,
)

from typing import Optional


class ConfigManager:

    def __init__(self, function_name: str, boto_session: boto3.Session):
        self._function_name = function_name
        self._initial_config = None
        self._aws_lambda_client = boto_session.client("lambda")
        self._aws_quotas_client = boto_session.client("service-quotas")

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

    def get_config(self):
        try:
            config = self._aws_lambda_client.get_function_configuration(
                FunctionName=self._function_name
            )

        except ParamValidationError as e:
            logger.debug(e.args[0])
            raise FunctionConfigurationError(e.args[0])

        except ClientError as e:
            raise FunctionConfigurationError(e.args[0])

        else:
            return FunctionConfig(
                memory_mb=config["MemorySize"], timeout_s=config["Timeout"]
            )

    def set_config(
        self,
        memory_mb: Optional[int] = None,
        timeout_s: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        try:
            if model_name == "None":
                model_name = None

            config = self._aws_lambda_client.get_function_configuration(
                FunctionName=self._function_name
            )
            current_env = config.get("Environment", {}).get("Variables", {})

            if memory_mb is None:
                memory_mb = int(config["MemorySize"])

            if not self._initial_config:
                self._initial_config = FunctionConfig(
                    memory_mb=config["MemorySize"],
                    timeout_s=config["Timeout"],
                    model_name=current_env.get("MODEL_NAME", None),
                )

            if model_name is not None:
                current_env["MODEL_NAME"] = model_name

            timeout = self._get_max_timeout() if timeout_s is None else timeout_s

            self._aws_lambda_client.update_function_configuration(
                FunctionName=self._function_name,
                MemorySize=int(memory_mb),
                Timeout=timeout,
                Environment={"Variables": current_env},
            )

            while (
                config["MemorySize"] != memory_mb
                or config["LastUpdateStatus"] == "InProgress"
                or current_env.get("MODEL_NAME", None) != model_name
            ):
                w = self._aws_lambda_client.get_waiter("function_updated")
                w.wait(FunctionName=self._function_name)

                config = self._aws_lambda_client.get_function_configuration(
                    FunctionName=self._function_name
                )

                current_env = config.get("Environment", {}).get("Variables", {})

        except ParamValidationError as e:
            logger.debug(e.args[0])
            raise FunctionConfigurationError(e.args[0])

        except ClientError as e:

            if e.response["Error"]["Code"] == "ResourceConflictException":
                logger.warning("Concurrent Update Function Error. Retrying ...")

                time.sleep(2)

                self.set_config(
                    memory_mb=memory_mb, timeout_s=timeout_s, model_name=model_name
                )

            else:
                raise FunctionConfigurationError(e.args[0])

        else:
            return config

    def reset_config(self):
        if self._initial_config is None:
            raise FunctionConfigurationError("Initial configuration not set.")
        self.set_config(
            memory_mb=self._initial_config.memory_mb,
            timeout_s=self._initial_config.timeout_s,
            model_name=self._initial_config.model_name,
        )
