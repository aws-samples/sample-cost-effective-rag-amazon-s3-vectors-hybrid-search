"""Cached boto3 clients with adaptive retries.

All modules access AWS through this single seam (``aws_clients.xyz()``),
which keeps unit tests trivial: patch here, everything downstream is covered.

Adaptive retry mode is the documented mitigation for Bedrock
ThrottlingException and S3 Vectors TooManyRequestsException.
"""
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig

_RETRY_CONFIG = BotoConfig(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=60,
)


@lru_cache(maxsize=1)
def s3():
    return boto3.client("s3", config=_RETRY_CONFIG)


@lru_cache(maxsize=1)
def s3vectors():
    return boto3.client("s3vectors", config=_RETRY_CONFIG)


@lru_cache(maxsize=1)
def bedrock_runtime():
    return boto3.client("bedrock-runtime", config=_RETRY_CONFIG)


@lru_cache(maxsize=4)
def dynamodb_table(table_name: str):
    return boto3.resource("dynamodb", config=_RETRY_CONFIG).Table(table_name)
