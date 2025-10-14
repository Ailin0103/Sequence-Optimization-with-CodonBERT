"""
This module provides functionality to connect to an RDS PostgreSQL database,
create schemas, create tables, and insert data into tables.

It includes:
- A function to establish a connection to the database.
- Functions to create schemas and tables.
- A function to insert data from a DataFrame into a table.

The module also includes type detection for DataFrame columns, supporting
INTEGER, TEXT, FLOAT, and TIMESTAMP types.
"""

from typing import Any

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection
from psycopg2.extras import execute_values

from sequence_optimization_with_codonbert.aws_auth import get_secret
from sequence_optimization_with_codonbert.config import (
    DATABASE_NAME,
    HOST_ENDPOINT,
    PORT,
    PROFILE_NAME,
    REGION_NAME,
    SECRET_NAME,
)


def connect_to_rds(
    host_endpoint: str, database_name: str, user: str, password: str, port: int = 5432
) -> connection:
    """
    Connects to an RDS PostgreSQL database using the provided credentials.

    Args:
        host_endpoint (str): The endpoint of the RDS PostgreSQL database.
        database_name (str): The name of the database to connect to.
        user (str): The username for authentication.
        password (str): The password for authentication.
        port (int, optional): The port number for the database. Defaults to 5432.

    Returns:
        connection: The database connection object.

    Raises:
        Exception: If there is an error connecting to the database.
    """
    try:
        conn = psycopg2.connect(
            host=host_endpoint,
            database=database_name,
            user=user,
            password=password,
            port=port,
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"Connected to database. PostgreSQL version: {db_version}")
        cur.close()
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise


def create_schema(connection: connection, schema_name: str) -> None:
    """
    Creates a schema in the database if it does not already exist.

    Args:
        connection (connection): The database connection object.
        schema_name (str): The name of the schema to create.

    Raises:
        Exception: If there is an error creating the schema.
    """
    with connection.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(schema_name)
            )
        )
        connection.commit()
        print(f"Schema '{schema_name}' created or already exists.")


def get_sql_type(dtype: Any) -> str:
    """
    Maps a pandas DataFrame column dtype to a PostgreSQL column type.

    Args:
        dtype (Any): The dtype of the DataFrame column.

    Returns:
        str: The corresponding PostgreSQL column type.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"


def create_table(
    connection: connection, df: pd.DataFrame, schema_name: str, table_name: str
) -> None:
    """
    Creates a table in the specified schema based on the DataFrame structure.

    Args:
        connection (connection): The database connection object.
        df (pd.DataFrame): The DataFrame whose structure will define the table.
        schema_name (str): The name of the schema where the table will be created.
        table_name (str): The name of the table to create.

    Raises:
        Exception: If there is an error creating the table.
    """
    with connection.cursor() as cur:
        columns = ", ".join(f"{col} {get_sql_type(df[col])}" for col in df.columns)
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.{} (
                {}
            )
        """).format(
            sql.Identifier(schema_name), sql.Identifier(table_name), sql.SQL(columns)
        )
        cur.execute(create_table_query)
        connection.commit()
        print(f"Table '{schema_name}.{table_name}' created or already exists.")


def insert_data(
    connection: connection, df: pd.DataFrame, schema_name: str, table_name: str
) -> None:
    """
    Inserts data from a DataFrame into the specified table.

    Args:
        connection (connection): The database connection object.
        df (pd.DataFrame): The DataFrame containing the data to insert.
        schema_name (str): The name of the schema containing the table.
        table_name (str): The name of the table to insert data into.

    Raises:
        Exception: If there is an error inserting the data.
    """
    with connection.cursor() as cur:
        for _, row in df.iterrows():
            insert_query = sql.SQL("""
                INSERT INTO {}.{} ({}) VALUES ({})
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(", ").join(map(sql.Identifier, df.columns)),
                sql.SQL(", ").join(sql.Placeholder() * len(df.columns)),
            )
            cur.execute(insert_query, tuple(row))
        connection.commit()
        print("Data inserted successfully.")


def insert_data_bulk(
    connection,
    df: pd.DataFrame,
    schema_name: str,
    table_name: str,
    page_size: int = 200,
) -> None:
    """
    Efficiently inserts data from a DataFrame into the specified table.

    Args:
        connection (connection): The database connection object.
        df (pd.DataFrame): The DataFrame containing the data to insert.
        schema_name (str): The name of the schema containing the table.
        table_name (str): The name of the table to insert data into.
        page_size (int): Number of rows to insert per batch (default: 200).

    Raises:
        Exception: If there is an error inserting the data.
    """
    cols = list(df.columns)

    insert_query = sql.SQL("""
        INSERT INTO {}.{} ({})
        VALUES %s
    """).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, cols)),
    )

    # Convert DataFrame to list of tuples
    data = [tuple(row) for row in df.to_numpy()]

    with connection.cursor() as cur:
        execute_values(cur, insert_query, data, page_size=page_size)
        connection.commit()

    print(f"Inserted {len(df)} rows into {schema_name}.{table_name} successfully.")


# Example usage
if __name__ == "__main__":
    credentials = get_secret(
        SECRET_NAME, region_name=REGION_NAME, profile_name=PROFILE_NAME
    )
    if credentials is None:
        raise ValueError("Failed to retrieve credentials from AWS Secrets Manager.")
    user = credentials["username"]
    password = credentials["password"]
    conn = connect_to_rds(HOST_ENDPOINT, DATABASE_NAME, user, password, PORT)

    data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    df = pd.DataFrame(data)
    SCHEMA_NAME = "example"
    TABLE_NAME = "sample_table"

    try:
        create_schema(conn, SCHEMA_NAME)
        create_table(conn, df, SCHEMA_NAME, TABLE_NAME)
        insert_data(conn, df, SCHEMA_NAME, TABLE_NAME)
    finally:
        conn.close()
