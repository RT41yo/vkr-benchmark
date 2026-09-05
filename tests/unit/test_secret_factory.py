from vkr_benchmark.inputs import create_secret_source


def test_secret_factory_reproduces_same_stream_for_same_secret_id() -> None:
    left = create_secret_source("000001").read_bits(64)
    right = create_secret_source("000001").read_bits(64)
    assert left == right


def test_secret_factory_separates_different_secret_ids() -> None:
    left = create_secret_source("000001").read_bits(64)
    right = create_secret_source("000002").read_bits(64)
    assert left != right
