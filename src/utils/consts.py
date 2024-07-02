import os
from dotenv import load_dotenv


load_dotenv()


MEM_OPTS = [128 + i * 192 for i in range(16)]
IAM = os.getenv('IAM')
LAMBDA_ROLE = os.getenv('LAMBDA_ROLE')