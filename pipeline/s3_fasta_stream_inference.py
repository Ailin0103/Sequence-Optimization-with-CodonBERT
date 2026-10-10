import boto3
import gzip
import io
import json
from datetime import datetime

# 建议显式指定 region（可选）
session = boto3.Session()  # 默认会用你的 AWS CLI / env / IAM role
s3 = session.client("s3")

