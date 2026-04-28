"""AWS Price List API adapter for Lambda pricing.

The Price List API is only served from ``us-east-1`` (and ``ap-south-1``); that
is the *endpoint* region and is independent of the region whose prices we want,
which is passed as a filter.

Two things this adapter must get right:

* **Pagination.** ``GetProducts`` returns at most 100 products per call and
  AWSLambda publishes far more than that (every region's duration, request,
  provisioned-concurrency, storage and edge SKUs share one service code). A
  single unpaginated call can therefore miss the very SKU being priced, and the
  failure surfaces as "could not parse pricing information" rather than as a
  wrong number — but only sometimes, depending on the page boundary.
* **Session reuse.** The client is built from the caller's session, so region,
  credentials, profile and any ``AWS_ENDPOINT_URL`` override apply here too.
  Previously this module called ``boto3.client`` directly, silently ignoring the
  session every other adapter was constructed with.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from optiserve.exceptions import CostCalculationError
from optiserve.logging import get_logger

logger = get_logger(__name__)

# Price List API endpoint region (not the priced region).
_PRICING_ENDPOINT_REGION = "us-east-1"

# Cap on pages walked, so a filter that matches an unexpectedly large catalogue
# cannot turn a pricing lookup into an unbounded paginated crawl.
_MAX_PAGES = 50


class PricingClient:
    """Fetches Lambda compute (GB-second) and request unit prices."""

    def __init__(
        self,
        endpoint_region: str = _PRICING_ENDPOINT_REGION,
        boto_session: boto3.Session | None = None,
    ) -> None:
        # Imported here to keep the module importable without the session
        # helper's botocore Config dependency at collection time.
        from optiserve.aws.session import create_client, create_session

        session = boto_session or create_session()
        self._client = create_client(session, "pricing", region_name=endpoint_region)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_lambda_pricing_units(
        self, region: str = "us-east-1", architecture: str = "x86_64"
    ) -> dict[str, float]:
        """Return ``{'compute': usd_per_gb_second, 'request': usd_per_request}``
        for the given region and architecture.

        Note: when a product exposes tiered prices we take the highest (on-demand,
        pre-free-tier) tier — a deliberately conservative choice for relative
        comparison between configurations.
        """
        duration_group, request_group = (
            ("AWS-Lambda-Duration-ARM", "AWS-Lambda-Requests-ARM")
            if architecture == "arm64"
            else ("AWS-Lambda-Duration", "AWS-Lambda-Requests")
        )

        wanted = {duration_group: "compute", request_group: "request"}
        units: dict[str, float] = {}

        for price in self._iter_price_list(region):
            for group, key in wanted.items():
                if key in units:
                    continue
                if self._matches_group(price, group):
                    price_value = self._extract_price(price)
                    if price_value is not None:
                        units[key] = price_value
            if len(units) == len(wanted):
                break

        missing = [group for group, key in wanted.items() if key not in units]
        if missing:
            raise CostCalculationError(
                "Could not parse pricing information for group(s) "
                f"{', '.join(missing)} in region {region!r}."
            )
        return units

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _iter_price_list(self, region: str) -> Iterator[str]:
        """Yield every product JSON string for AWSLambda in ``region``.

        Uses the paginator when botocore provides one and falls back to manual
        ``NextToken`` looping otherwise (some AWS-compatible endpoints do not
        register the paginator).
        """
        filters = [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}]
        try:
            if self._client.can_paginate("get_products"):
                paginator = self._client.get_paginator("get_products")
                pages = paginator.paginate(ServiceCode="AWSLambda", Filters=filters)
                for page_number, page in enumerate(pages, start=1):
                    yield from page.get("PriceList", [])
                    if page_number >= _MAX_PAGES:
                        logger.warning(
                            "Stopped paginating Lambda pricing after %d pages.", _MAX_PAGES
                        )
                        return
                return

            next_token: str | None = None
            for _page in range(_MAX_PAGES):
                kwargs: dict[str, Any] = {"ServiceCode": "AWSLambda", "Filters": filters}
                if next_token:
                    kwargs["NextToken"] = next_token
                response = self._client.get_products(**kwargs)
                yield from response.get("PriceList", [])
                next_token = response.get("NextToken")
                if not next_token:
                    return
            logger.warning("Stopped paginating Lambda pricing after %d pages.", _MAX_PAGES)
        except (ClientError, BotoCoreError) as exc:
            raise CostCalculationError(f"Error fetching Lambda pricing information: {exc}") from exc

    @staticmethod
    def _matches_group(price: Any, group: str) -> bool:
        """Whether a product belongs to ``group``.

        The Price List API returns each product as a JSON *string*. Parsing it
        is exact; the regex is kept only as a fallback for endpoints that hand
        back something json.loads cannot read.
        """
        document = PricingClient._as_document(price)
        if document is not None:
            attributes = document.get("product", {}).get("attributes", {})
            return attributes.get("group") == group
        return bool(re.search(rf'"group"\s*:\s*"{re.escape(group)}"', str(price)))

    @staticmethod
    def _extract_price(price: Any) -> float | None:
        """Highest on-demand USD unit price published for a product."""
        document = PricingClient._as_document(price)
        values: list[float] = []

        if document is not None:
            # AWS nests prices as terms.OnDemand.<sku>.priceDimensions.<rate>
            # .pricePerUnit.USD, but the depth has changed across API versions
            # and differs on AWS-compatible endpoints. Walk the OnDemand subtree
            # for any pricePerUnit.USD rather than hard-coding the level count.
            values.extend(_usd_values(document.get("terms", {}).get("OnDemand", {})))
        else:
            for match in re.findall(r'\{\s*"USD"\s*:\s*"([^"]*)"\s*\}', str(price)):
                with contextlib.suppress(ValueError):
                    values.append(float(match))

        return max(values) if values else None

    @staticmethod
    def _as_document(price: Any) -> dict | None:
        if isinstance(price, dict):
            return price
        try:
            parsed = json.loads(price)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    # Retained for backward compatibility: the previous implementation exposed
    # this static helper and a unit test pins its tier-selection behaviour.
    @staticmethod
    def _extract_group_price(price_list: Iterable[Any], group: str) -> float:
        for price in price_list:
            if PricingClient._matches_group(price, group):
                value = PricingClient._extract_price(price)
                if value is not None:
                    return value
        raise CostCalculationError(f"Could not parse pricing information for group '{group}'.")


def _as_float(value: Any) -> float | None:
    """Parse a price string, or ``None`` if it is not a number.

    A separate helper rather than ``contextlib.suppress`` inside the generator
    below: suppressing around a ``yield`` changes how the generator behaves when
    closed or thrown into, which is not the intent here.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usd_values(node: Any) -> Iterator[float]:
    """Every ``pricePerUnit.USD`` under ``node``, at any nesting depth."""
    if isinstance(node, dict):
        per_unit = node.get("pricePerUnit")
        if isinstance(per_unit, dict):
            price = _as_float(per_unit.get("USD"))
            if price is not None:
                yield price
        for value in node.values():
            yield from _usd_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _usd_values(value)
