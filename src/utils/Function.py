import uuid
import random
import string


class Function:
    def __init__(self,
                 code_dir,
                 runtime='python3.12',
                 name=f'Function_{"".join(random.choices(string.ascii_letters, k=8))}',
                 inputs=None,
                 outputs=None,
                 memory_size=128
                 ) -> None:

        self.id = uuid.uuid4().hex
        self.code_dir = code_dir
        self.runtime = runtime
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.memory_size = memory_size
        
        
    def __hash__(self) -> int:
        return hash(self.id)
    
    
    def __eq__(self, value: object) -> bool:
        return isinstance(value, Function) and self.id == value.id
        
        
    def __str__(self) -> str:
        return f"Function: {self.name} ID({self.id})"
    
    
    def get_number_of_versions(self) -> int:
        return len(self.code_dirs)