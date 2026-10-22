import stat
import textwrap

import pytest

from s1 import registry


@pytest.fixture
def fake_nvidia_smi(tmp_path):
    """An executable that prints two ticks of 2-GPU nvidia-smi rows (plus one [N/A] row), then idles."""
    script = tmp_path / "nvidia-smi"
    script.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            echo "0, 50, 1000, 23034, 60.0, 40"
            echo "1, 70, 3000, 23034, 80.0, 45"
            echo "0, 90, 2000, 23034, 70.0, 50"
            echo "1, 30, 4000, 23034, 50.0, 55"
            echo "0, [N/A], 2000, 23034, 70.0, 50"
            exec sleep 30
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return [str(script)]


class EchoModel:
    """Decision model that always picks the first option / says yes / picks level 0."""

    closed = False

    def decide(self, state, questions):
        ((key, q),) = questions.items()
        if q["type"] == "choice":
            opts = list(q["criteria"])
            probs = {o: (0.7 if i == 0 else 0.3 / (len(opts) - 1)) for i, o in enumerate(opts)}
            return {key: {"type": "choice", "choice": opts[0], "probabilities": probs}}
        if q["type"] == "noul":
            return {key: {"type": "noul", "noul": 0.8}}
        levels = {str(i): (1.0 if i == 0 else 0.0) for i in range(len(q["criteria"]))}
        return {key: {"type": "score", "score": 0.0, "probabilities": levels}}

    def close(self):
        EchoModel.closed = True


@pytest.fixture
def echo_model():
    registry.MODELS["echo"] = lambda settings: EchoModel()
    EchoModel.closed = False
    yield EchoModel
    registry.MODELS.pop("echo", None)
