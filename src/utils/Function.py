import uuid


class Function:
    def __init__(self,
                 name,
                 version=None,
                 inputs=None,
                 outputs=None,
                 code_dir=None,
                 memory=128 # MB
                 ) -> None:
        self.id = uuid.uuid4().hex
        self.name = name
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.code_dir = code_dir
        self.memory = memory
        
        
    def __hash__(self) -> int:
        return hash(self.id)
    
    
    def __eq__(self, value: object) -> bool:
        return isinstance(value, Function) and self.id == value.id
        
        
    def __str__(self) -> str:
        return f"Function: {self.name} v{self.version}"