import numpy as np

import re
import json
import boto3
from botocore.exceptions import ClientError

from src.exceptions import CostCalculationError
from typing import Optional, Union


class CostCalculator:
    def __init__(self, function_name: Optional[str] = None):
        self.aws_pricing_units = None
        self._function_name = function_name

    def calculate_cost(
        self,
        memory_mb: int,
        duration_ms: Union[float, np.ndarray],
        calculate_invocation_cost: bool = True,
    ):
        if not self.aws_pricing_units:
            self.aws_pricing_units = self._get_amazon_pricing_units()

        memory_gb = memory_mb / 1024.0
        duration_s = np.ceil(duration_ms) / 1000.0

        compute_cost = self.aws_pricing_units["compute"] * memory_gb * duration_s

        return compute_cost + (
            self.aws_pricing_units["request"] if calculate_invocation_cost else 0
        )

    def _get_amazon_pricing_units(
        self, region: str = "us-east-1", architecture: str = "x86_64"
    ):
        try:
            response = boto3.client("pricing", region_name="us-east-1").get_products(
                ServiceCode="AWSLambda",
                Filters=[
                    {
                        "Type": "TERM_MATCH",
                        "Field": "regionCode",
                        "Value": region,
                    },
                ],
            )

        except ClientError as e:
            raise CostCalculationError(
                f"Error has been raised while fetching the pricing information: {e}"
            )

        else:
            price_groups = (
                ["AWS-Lambda-Duration-ARM", "AWS-Lambda-Requests-ARM"]
                if architecture == "arm64"
                else ["AWS-Lambda-Duration", "AWS-Lambda-Requests"]
            )

            pricing_units = []

            for group in price_groups:
                for price in response["PriceList"]:
                    if re.search(f'"group"\s*:\s*"{group}"', price):  # type: ignore
                        all_results = re.findall('\{"USD"\s*:\s*"[.\d]*"}', price)  # type: ignore

                        if all_results:
                            prices_per_tier = map(
                                lambda element: float(json.loads(element)["USD"]),
                                all_results,
                            )
                            pricing_units.append(max(prices_per_tier))
                            break

            try:
                return {"compute": pricing_units[0], "request": pricing_units[1]}
            except:
                raise CostCalculationError("Could not parse the pricing information.")
