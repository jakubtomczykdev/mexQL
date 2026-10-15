from mexql.config import SamplingCfg
from mexql.inference.base import DummyBackend
from mexql.inference.hf_backend import HfBackend


def test_hf_finish_reason_detection():
    backend = HfBackend.__new__(HfBackend)  # no model load
    backend.stop_ids = {7, 9}
    assert backend._finish([1, 2, 7, 3], 10) == ([1, 2], "stop")
    assert backend._finish([1, 2, 3], 3) == ([1, 2, 3], "length")
    assert backend._finish([1, 2], 10) == ([1, 2], "stop")
    assert backend._finish([9], 10) == ([], "stop")


def test_dummy_backend_n_samples():
    out = DummyBackend(lambda p: f"echo:{p}").generate(["a", "b"], SamplingCfg(n=3), seed=0)
    assert len(out) == 2 and len(out[0]) == 3 and out[1][0].text == "echo:b"
