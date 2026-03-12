"""boto3 session helper.

Centralizes session creation so region/profile handling lives in one place and
every AWS adapter is constructed the same way.
"""
from __future__ import annotations

from typing import Optional

import boto3


def create_session(
    region_name: Optional[str] = None, profile_name: Optional[str] = None
) -> boto3.Session:
    """Create a boto3 Session. Credentials come from the standard AWS
    credential chain (environment, shared config, or instance role)."""
    return boto3.Session(region_name=region_name, profile_name=profile_name)
