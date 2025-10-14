# Sequence-Optimization-with-CodonBERT

## Installation

1. Install `pixi` package management tool:

    ``` bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

2. Install python environment using `pixi`:

    ``` bash
    pixi install
    ```

## Authentication to AWS

1. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

2. Login to your AWS account and create a profile with [AWS CLI login](https://docs.aws.amazon.com/signin/latest/userguide/command-line-sign-in.html).
  
3. Replace the `profile_name` in [`config.py`](https://github.com/Ailin0103/Sequence-Optimization-with-CodonBERT/blob/main/src/sequence_optimization_with_codonbert/config.py) with the profile you created.

## Connection to AWS RDS PostgresSQL instance

1. Example of creating a new schema, creating a new table and inserting a dataframe can be found in [this script](https://github.com/Ailin0103/Sequence-Optimization-with-CodonBERT/blob/main/notebooks/connect_to_rds_example.ipynb).
