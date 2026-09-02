from vkr_benchmark.randomness import ControlRandomSource, MethodRandomSource


def test_streams_are_deterministic_for_same_role_and_seed() -> None:
    a = MethodRandomSource(12345)
    b = MethodRandomSource(12345)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_method_and_control_are_separate_objects() -> None:
    method = MethodRandomSource(7)
    control = ControlRandomSource(7)

    # Equal seed may intentionally yield equal values; isolation means advancing
    # one stream does not advance the other.
    first_method = method.random()
    method.random()
    first_control = control.random()
    assert first_control == first_method
