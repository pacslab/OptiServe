import os
import zipfile


def zip_dir(dir_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), dir_path))
           
                
def unzip_dir(zip_path: str, dir_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(dir_path)
        
        
def delete_file(file_path: str):
    os.remove(file_path)
    
    
def get_zip_file_as_bytes(zip_path: str) -> bytes:
    with open(zip_path, 'rb') as file:
        return file.read()