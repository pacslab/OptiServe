from src.exceptions.sampling_error import SamplingError


class LogParsingError(SamplingError):
    def __init__(self):
        super().__init__("Error parsing log file.")
