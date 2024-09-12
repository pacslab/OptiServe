import os
import gzip
import zipfile
import tarfile


def zip_dir(dir_path: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), dir_path))
                

def gzip_dir(dir_path: str, gzip_path: str):
    with tarfile.open(f"{dir_path}.tar", "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))

    with open(f"{dir_path}.tar", 'rb') as f_in, gzip.open(gzip_path, 'wb') as f_out:
        f_out.writelines(f_in)

    os.remove(f"{dir_path}.tar")
           
                
def unzip_dir(zip_path: str, dir_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(dir_path)
        
        
def delete_file(file_path: str):
    os.remove(file_path)
    
    
def get_file_as_bytes(path: str) -> bytes:
    with open(path, 'rb') as file:
        return file.read()