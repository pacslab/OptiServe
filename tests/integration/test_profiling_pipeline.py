"""Integration: a full profiling run against a mocked AWS account.

This is the end-to-end path the framework exists for — configure a Lambda,
invoke it, parse its REPORT log, sample adaptively, fit a curve — exercised
with no real AWS and no live function.

moto implements Lambda's control plane and CloudWatch Logs, but not ``invoke``
(that needs Docker), so the data plane is replaced at exactly one seam: the
``lambda`` client's ``invoke``. Everything above it — ConfigManager, Invoker
retry/backoff, LogParser, Explorer, Sampler, the acquisition loop and the curve
fit — is the real code.
"""

import base64
import math

import pytest

from optiserve.config import ProfilingConfig
from optiserve.modeling.function_model import FunctionPerformanceModeling
from optiserve.observability import EventName
from optiserve.profiling.state import JsonCheckpointStore

pytestmark = pytest.mark.integration


class FakeLambdaRuntime:
    """A deterministic Lambda whose latency really does fall with memory.

    ``rt(m) = a0 + a1 * exp(-m / a2)``, i.e. the shape OptiServe fits — so a
    successful run must recover parameters close to these. Below
    ``min_memory_mb`` the function is killed by the platform and emits the OOM
    marker AWS actually emits.
    """

    def __init__(self, a0=200.0, a1=4000.0, a2=600.0, min_memory_mb=0):
        self.a0, self.a1, self.a2 = a0, a1, a2
        self.min_memory_mb = min_memory_mb
        self.invocations = 0

    def latency_at(self, memory_mb):
        return self.a0 + self.a1 * math.exp(-memory_mb / self.a2)

    def report_log(self, memory_mb):
        if memory_mb < self.min_memory_mb:
            # The marker AWS actually emits for an out-of-memory kill; note the
            # reported usage is clamped at the limit, not above it.
            return (
                "START RequestId: r\n"
                "RequestId: r Error: Runtime exited with error: signal: killed\n"
                "END RequestId: r\n"
                "REPORT RequestId: r\tDuration: 100.00 ms\tBilled Duration: 100 ms\t"
                f"Memory Size: {memory_mb} MB\tMax Memory Used: {memory_mb} MB\t"
            )
        duration = self.latency_at(memory_mb)
        return (
            "START RequestId: r\nEND RequestId: r\n"
            f"REPORT RequestId: r\tDuration: {duration:.2f} ms\t"
            f"Billed Duration: {duration:.0f} ms\t"
            f"Memory Size: {memory_mb} MB\tMax Memory Used: {memory_mb // 2} MB\t"
        )

    def install(self, monkeypatch, session, function_name):
        """Replace exactly one AWS operation: ``lambda:Invoke``.

        Patching ``BaseClient._make_api_call`` is the seam that works under
        moto — OptiServe builds its own clients, so patching a bound method on
        one client instance would miss them. Every other operation still goes to
        moto, so the control-plane path under test is the real one.
        """
        import botocore.client

        original = botocore.client.BaseClient._make_api_call
        control_plane = session.client("lambda")

        def make_api_call(client_self, operation_name, api_params):
            if operation_name != "Invoke":
                return original(client_self, operation_name, api_params)
            self.invocations += 1
            memory = control_plane.get_function_configuration(
                FunctionName=api_params["FunctionName"]
            )["MemorySize"]
            log = self.report_log(memory)
            return {
                "StatusCode": 200,
                "LogResult": base64.b64encode(log.encode()),
            }

        monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", make_api_call)


@pytest.fixture
def profiler_factory(session, deployed_function, monkeypatch):
    def build(runtime, **kwargs):
        deployed_function(name="inference", memory=512, timeout=30)
        runtime.install(monkeypatch, session, "inference")
        defaults = {
            "function_name": "inference",
            "memory_bounds": (128, 3008),
            "memory_space_step": 128,
            "profiling_iterations": 2,
            "max_total_sample_count": 8,
            "boto_session": session,
            "config": ProfilingConfig(noise_reduction=False),
        }
        defaults.update(kwargs)
        return FunctionPerformanceModeling(**defaults)

    return build


def test_profiling_recovers_the_latency_curve(profiler_factory):
    runtime = FakeLambdaRuntime(a0=200.0, a1=4000.0, a2=600.0)
    model = profiler_factory(runtime)

    with model.profiling_session():
        curve = model.get_performance_model()

    assert runtime.invocations > 0
    # The asymptote is the parameter the optimizer's cost model leans on most.
    assert curve(100_000) == pytest.approx(200.0, rel=0.15)
    # And the curve must be decreasing in memory, or `minimize` is meaningless.
    assert curve(3008) < curve(128)


def test_profiling_restores_the_function_it_mutated(session, profiler_factory):
    model = profiler_factory(FakeLambdaRuntime())
    lambda_client = session.client("lambda")
    before = lambda_client.get_function_configuration(FunctionName="inference")

    with model.profiling_session():
        model.run()

    after = lambda_client.get_function_configuration(FunctionName="inference")
    assert after["MemorySize"] == before["MemorySize"]
    assert after["Timeout"] == before["Timeout"]


def test_profiling_restores_even_when_the_run_fails(session, profiler_factory):
    model = profiler_factory(FakeLambdaRuntime())
    lambda_client = session.client("lambda")
    before = lambda_client.get_function_configuration(FunctionName="inference")["MemorySize"]

    with pytest.raises(RuntimeError), model.profiling_session():
        model.explorer.config_manager.set_config(memory_mb=3008)
        raise RuntimeError("analysis crashed")

    assert (
        lambda_client.get_function_configuration(FunctionName="inference")["MemorySize"] == before
    )


def test_oom_prunes_the_memory_floor(profiler_factory):
    """The platform kills the function below 1024 MB; the sampler must raise its
    floor past that rather than keep paying for failed invocations."""
    runtime = FakeLambdaRuntime(min_memory_mb=1024)
    model = profiler_factory(runtime)

    with model.profiling_session():
        model.run()

    space = model.explorer.memory_spaces["None"]
    assert space.min() >= 1024
    samples = model.sampler.explorations["None"]
    assert min(samples.memories) >= 1024


def test_an_interrupted_run_resumes_from_its_checkpoint(tmp_path, profiler_factory):
    store = JsonCheckpointStore(tmp_path)

    first = FakeLambdaRuntime()
    model = profiler_factory(first, checkpoint_store=store)
    with model.profiling_session():
        model.sampler.exploration_init("None")
    invocations_before = first.invocations
    samples_before = len(model.sampler.explorations["None"])
    assert invocations_before > 0

    second = FakeLambdaRuntime()
    resumed = profiler_factory(second, checkpoint_store=store)
    with resumed.profiling_session():
        resumed.sampler.exploration_init("None")

    assert second.invocations == 0, "resume must not repeat paid-for invocations"
    assert len(resumed.sampler.explorations["None"]) == samples_before


def test_a_run_emits_a_traceable_event_stream(profiler_factory, recorded_events):
    model = profiler_factory(FakeLambdaRuntime())
    with model.profiling_session():
        model.run()

    names = set(recorded_events.names())
    assert EventName.RUN_STARTED in names
    assert EventName.RUN_FINISHED in names
    assert EventName.CONFIG_APPLIED in names
    assert EventName.SAMPLE_RECORDED in names
    assert EventName.CONFIG_RESTORED in names

    sample = recorded_events.of(EventName.SAMPLE_RECORDED)[0]
    assert sample.attributes["function"] == "inference"
    assert sample.attributes["mean_duration_ms"] > 0


def test_multiple_model_variants_are_profiled_independently(profiler_factory):
    model = profiler_factory(
        FakeLambdaRuntime(),
        available_models=["resnet-18", "resnet-50"],
        memory_bounds=[(128, 3008), (128, 3008)],
    )
    with model.profiling_session():
        for variant in model.available_models:
            model.run(model_name=variant)

    assert set(model.sampler.explorations) == {"resnet-18", "resnet-50"}
    for variant in model.available_models:
        assert model.param_functions[variant].params is not None
