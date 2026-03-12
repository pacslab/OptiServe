"""AWS Price List API adapter for Lambda pricing.

The Price List API is only served from ``us-east-1`` (and ``ap-south-1``); that
is the *endpoint* region and is independent of the region whose prices we want,
which is passed as a filter.
"""
from __future__ import annotations

import json
import re
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from optiserve.exceptions import CostCalculationError

# Price List API endpoint region (not the priced region).
_PRICING_ENDPOINT_REGION = "us-east-1"


class PricingClient:
    """Fetches Lambda compute (GB-second) and request unit prices."""

    def __init__(self, endpoint_region: str = _PRICING_ENDPOINT_REGION):
        self._client = boto3.client("pricing", region_name=endpoint_region)

    def get_lambda_pricing_units(
        self, region: str = "us-east-1", architecture: str = "x86_64"
    ) -> Dict[str, float]:
        """Return ``{'compute': usd_per_gb_second, 'request': usd_per_request}``
        for the given region and architecture.

        Note: when a product exposes tiered prices we take the highest (on-demand,
        pre-free-tier) tier — a deliberately conservative choice for relative
        comparison between configurations.
        """
        try:
            response = self._client.get_products(
                ServiceCode="AWSLambda",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}
                ],
            )
        except ClientError as exc:
            raise CostCalculationError(
                f"Error fetching Lambda pricing information: {exc}"
            )

        duration_group, request_group = (
            ("AWS-Lambda-Duration-ARM", "AWS-Lambda-Requests-ARM")
            if architecture == "arm64"
            else ("AWS-Lambda-Duration", "AWS-Lambda-Requests")
        )

        units: Dict[str, float] = {}
        for key, group in (("compute", duration_group), ("request", request_group)):
            units[key] = self._extract_group_price(response["PriceList"], group)
        return units

    @staticmethod
    def _extract_group_price(price_list, group: str) -> float:
        group_pattern = re.compile(rf'"group"\s*:\s*"{re.escape(group)}"')
        usd_pattern = re.compile(r'\{"USD"\s*:\s*"[.\d]*"\}')
        for price in price_list:
            if group_pattern.search(price):
                usd_values = [
                    float(json.loads(match)["USD"])
                    for match in usd_pattern.findall(price)
                ]
                if usd_values:
                    return max(usd_values)
        raise CostCalculationError(
            f"Could not parse pricing information for group '{group}'."
        )
