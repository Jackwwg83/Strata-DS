"""CPU tests for the I/O pieces of proto/ds41_proto.py, on small synthetic safetensors files.

These check byte offsets and dequantization math, which fail silently (wrong numbers, no error)
if they are off. Run: python -m pytest proto/tests -q
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

import torch_kernels  # noqa: E402

sys.modules["kernel"] = torch_kernels
import ds41_proto as P  # noqa: E402
import model as M  # noqa: E402


def _e8m0(vals):
    return torch.tensor(vals, dtype=torch.float32).to(torch.float8_e8m0fnu)


def test_lazy_engram_matches_official_embedding(tmp_path):
    n, d = 1000, 256
    w = (torch.randn(n, d) * 4).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = _e8m0(2.0 ** torch.randint(-8, 3, (n, d // 32)).float())
    path = tmp_path / "engram-layer-01.safetensors"
    save_file({"layers.1.engram.embed.scale": s, "layers.1.engram.embed.weight": w}, str(path))

    lazy = P.LazyEngramEmbedding(n, d)
    lazy.device = "cpu"
    lazy.bind(P.SafetensorsFile(str(path)), 1)
    ref = M.ParallelEngramEmbedding(n, d)
    with torch.no_grad():
        ref.weight.copy_(w)
        ref.scale.copy_(s)
    idx = torch.randint(0, n, (1, 37, 24))
    idx[0, 0, :3] = torch.tensor([0, n - 1, 5])
    assert torch.equal(lazy(idx), ref(idx))


def _write_fake_experts(tmp_path, layers=(0, 3), n_exp=4):
    """Two shards with the real EXL3 tensor shapes; expert tensors split across the shards."""
    shapes = {"w1": ([320, 144, 48], [5120], [2304]), "w3": ([320, 144, 48], [5120], [2304]),
              "w2": ([144, 320, 48], [2304], [5120])}
    shards, index, truth = [{}, {}], {}, {}
    for L in layers:
        for e in range(n_exp):
            for w, (tr, su, sv) in shapes.items():
                base = f"layers.{L}.ffn.experts.{e}.{w}."
                t = {"trellis": torch.randint(-32768, 32767, tr, dtype=torch.int16),
                     "suh": torch.randn(su).half(), "svh": torch.randn(sv).half(),
                     "mul1": torch.tensor(-2082680531, dtype=torch.int32)}
                for part, v in t.items():
                    k = (e + len(part)) % 2
                    shards[k][base + part] = v
                    index[base + part] = f"model-0000{k + 1}.safetensors"
                    truth[base + part] = v
    for k in range(2):
        save_file(shards[k], str(tmp_path / f"model-0000{k + 1}.safetensors"))
    json.dump({"weight_map": index}, open(tmp_path / "model.safetensors.index.json", "w"))
    return truth


def test_expert_store_reads_exact_bytes(tmp_path):
    truth = _write_fake_experts(tmp_path)
    ckpt = P.Checkpoint(str(tmp_path))
    store = P.ExpertStore(ckpt, threads=4, slots=4, pin=False)
    assert store.slot_bytes == 13_316_352            # 12 components, each 256 B aligned
    for L, e in ((0, 2), (3, 1), (3, 3)):
        slot, _ = store._read(L, e)
        for w, p, dt, shape, off, nb in store.layout:
            got = slot[off:off + nb].view(P.ST_DTYPES[dt]).view(shape)
            assert torch.equal(got, truth[f"layers.{L}.ffn.experts.{e}.{w}.{p}"]), (L, e, w, p)
        store.free.put(slot)


def test_checkpoint_view_and_wo_a_dequant(tmp_path):
    w = (torch.randn(64, 96) * 10).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = _e8m0(2.0 ** torch.randint(-6, 2, (2, 3)).float())
    save_file({"layers.0.attn.wo_a.weight": w, "layers.0.attn.wo_a.scale": s}, str(tmp_path / "a.safetensors"))
    json.dump({"weight_map": {"layers.0.attn.wo_a.weight": "a.safetensors",
                              "layers.0.attn.wo_a.scale": "a.safetensors"}},
              open(tmp_path / "model.safetensors.index.json", "w"))
    ckpt = P.Checkpoint(str(tmp_path))
    assert torch.equal(ckpt.get("layers.0.attn.wo_a.weight").float(), w.float())
    # the convert.py formula, as load_dense applies it
    wf = ckpt.get("layers.0.attn.wo_a.weight").float()
    sf = ckpt.get("layers.0.attn.wo_a.scale").float()
    deq = (wf.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)) * sf[:, None, :, None]).flatten(2, 3).flatten(0, 1)
    want = w.float() * sf.repeat_interleave(32, 0).repeat_interleave(32, 1)
    assert torch.equal(deq, want)


def test_bf16_head_nll_matches_direct_cross_entropy():
    torch.manual_seed(0)
    head = P.Bf16Head(1000, 64)
    with torch.no_grad():
        head.weight.copy_(torch.randn(1000, 64).bfloat16())
    x = torch.randn(1, 9, 64).bfloat16()
    tgt = torch.randint(0, 1000, (8,))
    P.STATE["nll_targets"] = tgt
    last = head(x)
    P.STATE["nll_targets"] = None
    logits = x[0, :-1].float() @ head.weight.float().t()
    want = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
    assert torch.allclose(P.STATE["nll"], want, atol=1e-4)
    assert torch.allclose(last[0], x[0, -1].float() @ head.weight.float().t(), atol=1e-4)


def test_expert_store_slots_stay_on_cpu_when_default_device_changes(tmp_path):
    """Regression: build_model sets the default device to cuda before creating the store."""
    _write_fake_experts(tmp_path, layers=(0,), n_exp=1)
    ckpt = P.Checkpoint(str(tmp_path))
    torch.set_default_device("meta")
    try:
        store = P.ExpertStore(ckpt, threads=1, slots=2, pin=False)
    finally:
        torch.set_default_device("cpu")
    slot = store.free.get()
    assert slot.device.type == "cpu"


def test_cpu_experts_output_stays_on_cpu_when_default_device_changes(monkeypatch):
    """Regression: build_model sets the default device to cuda; the CPU kernel needs CPU buffers."""
    seen = {}

    class FakeExt:
        @staticmethod
        def exl3_moe_cpu_forward(handle, x, sel, w, out, threads):
            seen["devices"] = {t.device.type for t in (x, sel, w, out)}

    import types
    fake_mod = types.SimpleNamespace(exllamav3_ext=FakeExt)
    monkeypatch.setitem(sys.modules, "exllamav3", types.SimpleNamespace(ext=fake_mod))
    monkeypatch.setitem(sys.modules, "exllamav3.ext", fake_mod)
    ce = P.CpuExperts(ckpt=None)
    ce.handles[0] = 0
    torch.set_default_device("meta")
    try:
        x = torch.zeros(1, 8, dtype=torch.float16, device="cpu")
        ce.forward(0, x, torch.zeros(1, 6, dtype=torch.int64, device="cpu"), torch.zeros(1, 6, device="cpu"))
    finally:
        torch.set_default_device("cpu")
    assert seen["devices"] == {"cpu"}
