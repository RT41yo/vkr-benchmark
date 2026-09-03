from vkr_benchmark.randomness import RecordingSecretSource, Shake256SecretSource


def test_recording_secret_source_is_transparent_and_records_consumption() -> None:
    base = Shake256SecretSource("recording-test")
    recording = RecordingSecretSource(base)

    first = recording.read_bits(3)
    second = recording.read_bits(5)

    expected = Shake256SecretSource("recording-test").read_bits(8)
    assert first + second == expected
    assert recording.consumed_bits == expected
    assert recording.position == 8
    assert base.position == 8
