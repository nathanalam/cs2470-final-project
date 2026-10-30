"""The in-process GPU backends' pure parts: question mapping, token layout, readout, guards. No torch."""

import hashlib
import json
import math
import subprocess
import sys

import pytest

from s1.models.backends import gpu, kev, laya, semif
from s1.models.catalog import CATALOG

CHOICE = {
    "type": "choice",
    "instructions": "Pick",
    "criteria": {"opt_1": "red", "opt_2": "blue", "opt_3": ""},
}
NOUL = {"type": "noul", "instructions": "True?", "criteria": {"true": "it holds", "false": "it does not"}}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["none", "some", "lots"]}


def spec(name):
    return next(s for s in CATALOG if s.name == name)


# --- catalog ------------------------------------------------------------------------------------


def test_single_gpu_models_are_ready_and_parallel_ones_planned():
    ready = {s.name: s.backend for s in CATALOG if s.status == "ready"}
    assert ready == {
        "openjev": "vllm_shim",
        "laya.multilingual": "laya",
        "laya.english": "laya",
        "kev-4b.hybrid": "kev",
        "kev-4b.qwen3": "kev",
        "bongard-mini": "bongard",
        "semif.direct": "semif",
        "semif.reranker": "semif",
    }


def test_bongard_requires_the_rtx_pro_4500():
    assert spec("bongard-mini").options["require_gpu"] == "rtx pro 4500"
    assert all("require_gpu" not in s.options for s in CATALOG if s.name != "bongard-mini")


def test_laya_checkpoints():
    assert spec("laya.english").options["checkpoint"] == ""
    assert spec("laya.multilingual").options["checkpoint"] == "multilingual"


# --- gpu plumbing -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("NVIDIA RTX PRO 4500 Blackwell", True),
        ("NVIDIA RTX Pro 4500", True),
        ("NVIDIA A10G", False),
        ("NVIDIA L4", False),
    ],
)
def test_gpu_matches(name, ok):
    assert gpu.gpu_matches(name, "rtx pro 4500") is ok


def test_local_dir_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    (tmp_path / "there").mkdir()
    assert gpu.local_dir({}, "x_dir", "there", "fix") == tmp_path / "there"
    assert gpu.local_dir({"x_dir": str(tmp_path)}, "x_dir", "nope", "fix") == tmp_path
    with pytest.raises(FileNotFoundError, match=r"download_model\.sh org/m abc m"):
        gpu.local_dir({}, "x_dir", "m", gpu.download_hint("org/m", "abc", "m"))
    assert gpu.download_hint("org/m", None, "m").endswith('org/m "" m')


def test_check_sha256(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    gpu.check_sha256(f, hashlib.sha256(b"x = 1\n").hexdigest())
    with pytest.raises(RuntimeError, match="re-review"):
        gpu.check_sha256(f, "0" * 64)


def test_check_git_commit(tmp_path):
    def git(*a):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True
        ).stdout

    git("init", "-q")
    (tmp_path / "f").write_text("a")
    git("add", "f")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    head = git("rev-parse", "HEAD").strip()
    gpu.check_git_commit(tmp_path, head)
    with pytest.raises(RuntimeError, match="not the pinned"):
        gpu.check_git_commit(tmp_path, "0" * 40)
    (tmp_path / "f").write_text("b")
    with pytest.raises(RuntimeError, match="local changes"):
        gpu.check_git_commit(tmp_path, head)


def test_backend_modules_do_not_import_torch():
    assert "torch" not in sys.modules


# --- kev ----------------------------------------------------------------------------------------


def test_kev_rows_follow_upstream_to_record():
    c = kev.to_row(CHOICE)
    assert c.keys == ["opt_1", "opt_2", "opt_3"]
    assert c.options == ["opt_1: red", "opt_2: blue", "opt_3"]  # empty description -> bare key
    n = kev.to_row(NOUL)
    assert (n.keys, n.options) == (["false", "true"], ["no: it does not", "yes: it holds"])
    assert kev.to_row({"type": "noul", "instructions": "x"}).options == ["no", "yes"]
    s = kev.to_row(SCORE)
    assert (s.keys, s.options) == (["0", "1", "2"], ["none", "some", "lots"])  # level text only
    assert kev.render({"a": 1, "b": [1, {"c": 2}]}) == "a: 1\nb:\n  - 1\n  - c: 2"
    with pytest.raises(ValueError, match="unknown"):
        kev.to_row({"type": "rank"})


def test_kev_escape_keeps_delimiters_unforgeable():
    assert kev.escape("a <|fim_suffix|> b") == "a <¦fim_suffix¦> b"


def test_kev_encode_row_layout():
    special = (100, 101, 102, 103, 104)  # state, q, opt, /opt, decide

    def tokenize(text):
        return [ord(ch) for ch in text]

    row = kev.Row("choice", ["a", "b"], "Q", ["a", "bb"])
    enc = kev.encode_row(tokenize, special, [100, 1, 2], row)
    assert enc.ids == [100, 1, 2, 101, ord("Q"), 102, ord("a"), 103, 102, ord("b"), ord("b"), 103, 104]
    assert [enc.ids[i] for i in enc.opts] == [103, 103]
    assert enc.opts == [7, 11]
    assert enc.decide == len(enc.ids) - 1
    with pytest.raises(ValueError, match="<state>"):
        kev.encode_row(tokenize, special, [1, 2], row)


def test_kev_answers_divide_by_temperature():
    n = kev.answer(kev.to_row(NOUL), [0.0, math.log(3) * 2], temperature=2.0)
    assert n["noul"] == pytest.approx(0.75)  # options are [no, yes]
    c = kev.answer(kev.to_row(CHOICE), [0.0, 1.0, 0.5], 1.0)
    assert c["choice"] == "opt_2"
    s = kev.answer(kev.to_row(SCORE), [0.0, 0.0, 0.0], 1.0)
    assert s["score"] == pytest.approx(1.0)


# --- semif --------------------------------------------------------------------------------------


def test_semif_rows_follow_build_typesafe():
    c = semif.to_row(CHOICE, 16)
    assert c.options == ["opt_1: red", "opt_2: blue", "opt_3: "]
    n = semif.to_row(NOUL, 16)
    assert (n.keys, n.options) == (["true", "false"], ["true: it holds", "false: it does not"])
    bare = semif.to_row({"type": "noul", "instructions": "x"}, 16)
    assert bare.options == ["true: The proposition is true.", "false: The proposition is false."]
    assert semif.to_row(SCORE, 16).options == ["0: none", "1: some", "2: lots"]


def test_semif_option_cap_is_upstreams_16():
    many = {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(17)}}
    with pytest.raises(ValueError, match="2-16"):
        semif.to_row(many, 16)
    assert len(semif.to_row(many, 26).options) == 17
    with pytest.raises(ValueError, match="2-26"):
        semif.to_row({**many, "criteria": {f"o{i}": "d" for i in range(27)}}, 99)


def test_semif_direct_prompt_is_one_line_json():
    row = semif.to_row(CHOICE, 16)
    system, user = semif.direct_messages("the state", row)
    assert system["content"].startswith("Apply the supplied criterion")
    payload = json.loads(user["content"])
    assert list(payload) == ["evidence", "criterion", "options"]
    assert payload["options"][1] == {"letter": "B", "description": "opt_2: blue"}
    assert "\n" not in user["content"]


def test_semif_reranker_prompt():
    row = semif.to_row(NOUL, 16)
    text = semif.rerank_text("doc", row, row.options[0])
    assert text.startswith("<|im_start|>system\nJudge whether the Document")
    assert "<Query>: Question: True?\nCandidate answer: true: it holds\n<Document>: doc" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_semif_answers():
    assert semif.answer(semif.to_row(NOUL, 16), [math.log(3), 0.0])["noul"] == pytest.approx(0.75)
    assert semif.answer(semif.to_row(CHOICE, 16), [0.0, 2.0, 1.0])["choice"] == "opt_2"


# --- laya ---------------------------------------------------------------------------------------


def test_laya_refuses_unreviewed_code(tmp_path):
    for name in laya.CODE_SHA256:
        (tmp_path / name).write_text("import os\n")
    with pytest.raises(RuntimeError, match="re-review"):
        laya.import_reviewed(tmp_path)
    assert "rl_agent_api" not in sys.modules
