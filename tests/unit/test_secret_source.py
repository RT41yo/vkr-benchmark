from vkr_benchmark.randomness import Shake256SecretSource


def test_shake256_secret_source_has_stable_vector() -> None:
    source = Shake256SecretSource("000001")
    assert source.read_bits(24) == (
        1, 1, 0, 1, 0, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 0,
    )
    assert source.position == 24


def test_secret_reads_are_streaming_not_restarted() -> None:
    one = Shake256SecretSource("same")
    two = Shake256SecretSource("same")

    first = one.read_bits(7)
    second = one.read_bits(13)
    combined = two.read_bits(20)

    assert first + second == combined
