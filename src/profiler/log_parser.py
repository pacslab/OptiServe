import re


from src.exceptions import (
    NotEnoughMemory,
    FunctionTimeout,
    LogParsingError,
    InvocationError,
)

from src.utils.logger import logger

class LogParser:
    def __init__(self):
        self.log_parsing_params = [
            'Duration',
            'Billed Duration',
            'Max Memory Used',
            'Memory Size',
            'Init Duration',
        ]
        
        
    def parse(self, log: str):
        results = {}
        for key in self.log_parsing_params:
            match = re.match(rf".*\\t{key}: (?P<value>[0-9.]+) (ms|MB).*", log)
            if match:
                results[key] = float(match["value"])

        if "Billed Duration" not in results:
            raise LogParsingError()

        exec_time_ms = int(results["Billed Duration"])

        logger.info(f'Invocation Results: {results}')

        if "Task timed out after" in log:
            raise FunctionTimeout()


        if results["Max Memory Used"] > results["Memory Size"]:
            raise NotEnoughMemory(duration_ms=exec_time_ms)

        error_msg = re.match(r".*\[ERROR\] (?P<error>.*)END RequestId.*", log)
        if error_msg is not None:
            raise InvocationError(duration_ms=exec_time_ms, message=error_msg["error"])

        return exec_time_ms