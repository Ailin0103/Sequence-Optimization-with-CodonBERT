"""
This module contains configuration constants for the Revenue Forecasting project.

It includes AWS Secrets Manager configuration details and RDS PostgreSQL database
connection parameters.
"""

# AWS Secrets Manager configuration
SECRET_NAME = "rds!cluster-e75ce108-6d41-4717-9ff5-9303d9eb1cf5"
REGION_NAME = "us-east-1"
PROFILE_NAME = "ailin-codonbert"

# RDS PostgreSQL configuration
HOST_ENDPOINT = "database-1-instance-1.c0vqum0a4rl7.us-east-1.rds.amazonaws.com"
DATABASE_NAME = "postgres"
PORT = 5432
