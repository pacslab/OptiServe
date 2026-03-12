"""AWS Lambda cost model.

Computes the cost of an invocation from GB-second compute pricing plus a
per-request fee, using unit prices fetched (lazily, then cached) from the AWS
Price List API via :class:`~optiserve.aws.pricing_client.PricingClient`.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from optiserve.aws.pricing_client import PricingClient


class CostCalculator:
    """Lambda cost = compute_price * memory_gb * ceil(duration_ms)/1000 +
    request_price.

    Duration is billed at 1 ms granularity (``ceil``). Pricing units are fetched
    once on first use and cached for the lifetime of the instance.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        architecture: str = "x86_64",
        pricing_client: Optional[PricingClient] = None,
    ):
        self._region = region
        self._architecture = architecture
        self._pricing_client = pricing_client
        self.aws_pricing_units: Optional[dict] = None

    def _units(self) -> dict:
        if self.aws_pricing_units is None:
            client = self._pricing_client or PricingClient()
            self.aws_pricing_units = client.get_lambda_pricing_units(
                region=self._region, architecture=self._architecture
            )
        return self.aws_pricing_units

    def calculate_cost(
        self,
        memory_mb: int,
        duration_ms: Union[float, np.ndarray],
        calculate_invocation_cost: bool = True,
    ) -> Union[float, np.ndarray]:
        units = self._units()
        memory_gb = memory_mb / 1024.0
        duration_s = np.ceil(duration_ms) / 1000.0
        compute_cost = units["compute"] * memory_gb * duration_s
        request_cost = units["request"] if calculate_invocation_cost else 0.0
        return compute_cost + request_cost
