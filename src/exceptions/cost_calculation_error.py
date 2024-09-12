class CostCalculationError(Exception):
    def __initn__(self, message: str = 'Error in cost calculation.'):
        super().__init__(self.message)