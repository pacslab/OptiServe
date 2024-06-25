class PerformanceMetrics:
    def __init__(self, applicationId=None, functionId=None,
                 memory_usage=None, cpu_usage=None, response_time=None, accuracy=None):
        self.applicationId = applicationId
        self.functionId = functionId
        self.memory_usage = memory_usage
        self.cpu_usage = cpu_usage
        self.response_time = response_time
        self.accuracy = accuracy