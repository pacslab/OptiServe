class FunctionConfigurationError(Exception):
    def __init__(self, message: str = 'Error in function configuration. Make sure the provided function exists, and the configuration parameters are correct.'):
        super().__init__(self.message)