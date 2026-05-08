"""
This module provides functionality to retrieve secrets from AWS Secrets Manager.

It includes a function to fetch secrets using the AWS SDK (boto3) and handles
both string and binary secret formats. The module also demonstrates an example
usage of the `get_secret` function.
"""

import json
from typing import Any, Dict, Optional

from boto3.session import Session
from botocore.exceptions import ClientError

from sequence_optimization_with_codonbert.config import (
    PROFILE_NAME,
    REGION_NAME,
    SECRET_NAME,
)


def get_secret(
    secret_name: str, region_name: str, profile_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a secret from AWS Secrets Manager.

    This function connects to AWS Secrets Manager using the provided region and
    optional profile name, retrieves the secret value, and parses it into a dictionary.

    Args:
        secret_name (str): The name or ARN of the secret to retrieve.
        region_name (str): The AWS region where the secret is stored.
        profile_name (Optional[str], optional): The name of the AWS CLI profile to use. Defaults to None.

    Returns:
        Optional[Dict[str, Any]]: The secret as a dictionary if retrieval is successful, or None if an error occurs.

    Raises:
        ClientError: If there is an error with the AWS Secrets Manager client.
    """
    # Create a Secrets Manager client
    #session = Session(profile_name=profile_name)
    session = Session(profile_name=PROFILE_NAME)
    client = session.client(service_name="secretsmanager", region_name=region_name)
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        # Retrieve the secret value
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)

        # Parse and return the secret
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            return json.loads(secret)  # Return as a dictionary
        else:
            # If the secret is stored as binary
            secret = get_secret_value_response["SecretBinary"]
            return json.loads(secret.decode("utf-8"))
    except ClientError as e:
        print(f"Error retrieving secret: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    secret = get_secret(SECRET_NAME, REGION_NAME, profile_name=PROFILE_NAME)

    if secret:
        print("Retrieved secret:", secret)
