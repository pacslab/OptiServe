class PerformanceMetrics:
    def __init__(self, applicationId=None, functionId=None, max_memory_usage=None,
                 memory_usage=None, billable_duration=None, duration=None, init_duration=None, cpu_usage=None, response_time=None, accuracy=None, invocation_time=None):
        self.applicationId = applicationId
        self.functionId = functionId
        self.max_memory_usage = max_memory_usage
        self.memory_usage = memory_usage
        self.billable_duration = billable_duration
        self.duration = duration
        self.init_duration = init_duration
        self.cpu_usage = cpu_usage
        self.response_time = response_time
        self.accuracy = accuracy
        self.invocation_time = invocation_time
        
    def __str__(self):
        return f"PerformanceMetrics(applicationId={self.applicationId}, functionId={self.functionId}, max_memory_usage={self.max_memory_usage}, memory_usage={self.memory_usage}, billable_duration={self.billable_duration}, duration={self.duration}, init_duration={self.init_duration}, cpu_usage={self.cpu_usage}, response_time={self.response_time}, accuracy={self.accuracy}, invocation_time={self.invocation_time})"
    
    
    def __repr__(self):
        return self.__str__()