"""Unit tests for the aws/ adapter layer and cost model (no live AWS)."""
from unittest.mock import Mock

import numpy as np
import pytest

from optiserve.aws.log_parser import LogParser
from optiserve.aws.logs_client import AWSFunctionLogs
from optiserve.aws.pricing_client import PricingClient
from optiserve.cost import CostCalculator
from optiserve.exceptions import FunctionTimeout, NotEnoughMemory


# --------------------------------------------------------------------------- #
# LogParser
# --------------------------------------------------------------------------- #
_OK_LOG = (
    "REPORT RequestId: abc\tDuration: 120.00 ms\tBilled Duration: 120 ms\t"
    "Memory Size: 512 MB\tMax Memory Used: 100 MB\t"
)
_OOM_LOG = (
    "REPORT RequestId: abc\tDuration: 120.00 ms\tBilled Duration: 120 ms\t"
    "Memory Size: 512 MB\tMax Memory Used: 600 MB\t"
)
_TIMEOUT_LOG = (
    "2024-01-01 Task timed out after 3.00 seconds\n"
    "REPORT RequestId: abc\tBilled Duration: 3000 ms\tMemory Size: 512 MB\t"
    "Max Memory Used: 100 MB\t"
)


def test_logparser_returns_billed_duration():
    assert LogParser().parse_function_execution_time(_OK_LOG) == 120.0


def test_logparser_propagates_oom():
    # Regression: OOM must NOT be swallowed (sampler relies on this to prune).
    with pytest.raises(NotEnoughMemory):
        LogParser().parse_function_execution_time(_OOM_LOG)


def test_logparser_propagates_timeout():
    with pytest.raises(FunctionTimeout):
        LogParser().parse_function_execution_time(_TIMEOUT_LOG)


# --------------------------------------------------------------------------- #
# PricingClient tier parsing (static, no client)
# --------------------------------------------------------------------------- #
def test_pricing_extract_picks_max_tier():
    price_list = [
        '{"product": {"attributes": {"group": "AWS-Lambda-Duration"}}, '
        '"terms": {"OnDemand": {"x": {"y": {"pricePerUnit": {"USD": "0.0000166667"}}}, '
        '"z": {"w": {"pricePerUnit": {"USD": "0.0000200000"}}}}}}'
    ]
    price = PricingClient._extract_group_price(price_list, "AWS-Lambda-Duration")
    assert price == pytest.approx(0.00002)


# --------------------------------------------------------------------------- #
# CostCalculator formula (injected pricing units — no AWS)
# --------------------------------------------------------------------------- #
def test_cost_formula_scalar_and_vector():
    fake = Mock()
    fake.get_lambda_pricing_units.return_value = {"compute": 1e-5, "request": 2e-7}
    calc = CostCalculator(pricing_client=fake)

    # 1024 MB (=1 GB) for 1000 ms (=1 s): compute = 1e-5*1*1 = 1e-5, +req 2e-7.
    assert calc.calculate_cost(1024, 1000) == pytest.approx(1e-5 + 2e-7)
    assert calc.calculate_cost(1024, 1000, calculate_invocation_cost=False) == pytest.approx(1e-5)

    out = calc.calculate_cost(1024, np.array([1000.0, 2000.0]))
    assert np.allclose(out, [1e-5 + 2e-7, 2e-5 + 2e-7])
    # units fetched once and cached
    assert fake.get_lambda_pricing_units.call_count == 1


# --------------------------------------------------------------------------- #
# AWSFunctionLogs.get_logs_df concat fix
# --------------------------------------------------------------------------- #
def test_get_logs_df_concatenates_all_models():
    logs = AWSFunctionLogs(boto_session=Mock(), function_name="f")
    logs.get_logs = lambda start_time, end_time: {  # type: ignore
        "m1": [{"Duration": 1.0}],
        "m2": [{"Duration": 2.0}],
    }
    df = logs.get_logs_df(start_time=0, end_time=1)
    # Regression: previously iterated an empty frame and returned empty.
    assert set(df["Model"]) == {"m1", "m2"}
    assert len(df) == 2
