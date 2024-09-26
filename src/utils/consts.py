import os
from dotenv import load_dotenv


load_dotenv()


IAM = os.getenv('IAM')
LAMBDA_ROLE = os.getenv('LAMBDA_ROLE')
REGION = os.getenv('REGION')