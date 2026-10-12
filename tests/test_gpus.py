from s1cli import gpus


def test_parse_gres_typed():
    assert gpus.parse_gres("gpu:a10g:4(S:0)") == [("a10g", 4)]


def test_parse_gres_untyped():
    assert gpus.parse_gres("gpu:2") == [(None, 2)]
    assert gpus.parse_gres("gpu") == [(None, 1)]


def test_parse_gres_multi_and_null():
    assert gpus.parse_gres("gpu:a100:2(S:0-1),shard:8") == [("a100", 2)]
    assert gpus.parse_gres("(null)") == []
    assert gpus.parse_gres("") == []


def test_vram():
    assert gpus.vram_gb("a10g") == 24
    assert gpus.vram_gb("nvidia_a100-sxm4-80gb") == 80
    assert gpus.vram_gb("L40S") == 48
    assert gpus.vram_gb("mystery") is None


def test_guess_from_instance():
    assert gpus.guess_from_instance("gpu-dy-g5xlarge-1") == ("a10g", 24)
    assert gpus.guess_from_instance("dynamic,g6e.2xlarge") == ("l40s", 48)
    assert gpus.guess_from_instance("gpu-dy-g6xlarge-3") == ("l4", 24)
    assert gpus.guess_from_instance("cpu-dy-c5x2large-1") is None
