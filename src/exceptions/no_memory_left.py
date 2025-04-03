from src.exceptions.sampling_error import SamplingError


class NoMemoryLeft(SamplingError):
    def __init__(self):
        super().__init__("No memory left in the memory space to explore with.")
