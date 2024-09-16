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
        self.function_log_parsing_params = [
            'Duration',
            'Billed Duration',
            'Max Memory Used',
            'Memory Size',
            'Init Duration',
        ]
        
        
    def _get_function_invocation_logs(self, log: str):
        results = {}
        for key in self.function_log_parsing_params:
            match = re.match(rf".*\\t{key}: (?P<value>[0-9.]+) (ms|MB).*", log)
            if match:
                results[key] = float(match["value"])

        if "Billed Duration" not in results:
            raise LogParsingError()

        logger.info(f'Invocation Results: {results}')

        if "Task timed out after" in log:
            raise FunctionTimeout()

        if results["Max Memory Used"] > results["Memory Size"]:
            raise NotEnoughMemory(duration_ms=int(results["Billed Duration"]))

        error_msg = re.match(r".*\[ERROR\] (?P<error>.*)END RequestId.*", log)
        if error_msg is not None:
            raise InvocationError(duration_ms=int(results["Billed Duration"]), message=error_msg["error"])
        
        return results
        
        
    def parse_execution_time(self, log: str):
        results = self._get_function_invocation_logs(log)

        exec_time_ms = results["Billed Duration"]

        return exec_time_ms
    
    
    def parse_profiling_logs(self, log: str):
        results = {}
        for key in self.function_log_parsing_params:
            match = re.search(rf"{key}: (?P<value>[0-9.]+) (ms|MB)", log)
            if match:
                results[key] = float(match.group('value'))

        logger.info(f'Profiling Results: {results}')
        
        return results