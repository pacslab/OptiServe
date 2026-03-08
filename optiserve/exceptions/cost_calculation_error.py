class CostCalculationError(Exception):
    def __initn__(self, message: str = "Error in cost calculation."):
        self.message = message
        super().__init__(self.message)
