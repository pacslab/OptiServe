import os
from dotenv import load_dotenv


load_dotenv()


MODEL_MEM_OPTS = [128 + i * 64 for i in range(46)]
MEM_OPTS = [128 + i * 192 for i in range(16)]

IAM = os.getenv('IAM')
LAMBDA_ROLE = os.getenv('LAMBDA_ROLE')