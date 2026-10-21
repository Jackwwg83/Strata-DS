"""pack.py's Source on a merged index: a tensor comes from the shard the index names, even when another shard of the
source has a tensor of the same name (a hybrid of two quantizations). numpy only, no torch.

Run: python tools/ds41/test_pack_source.py
"""
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack as P  # noqa: E402


def write_safetensors(path, tensors):
    """tensors: {name: bytes} as U8 vectors"""
    hdr, data, off = {}, b"", 0
    for name, raw in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + data)


class MergedIndex(unittest.TestCase):
    def test_a_tensor_comes_from_the_shard_the_index_names(self):
        with tempfile.TemporaryDirectory() as d:
            # both shards hold "e.w1.trellis"; the index takes it from the one sorted FIRST, "b" from the other
            write_safetensors(os.path.join(d, "model-00001-of-00017.safetensors"), {"e.w1.trellis": b"\x01" * 8})
            write_safetensors(os.path.join(d, "model-00001-of-00043.safetensors"),
                              {"e.w1.trellis": b"\x02" * 8, "b": b"\x03" * 4})
            json.dump({"weight_map": {"e.w1.trellis": "model-00001-of-00017.safetensors",
                                      "b": "model-00001-of-00043.safetensors"}},
                      open(os.path.join(d, "model.safetensors.index.json"), "w"))
            src = P.Source(d)
            self.assertEqual(bytes(src.read("e.w1.trellis")), b"\x01" * 8)
            self.assertEqual(bytes(src.read("b")), b"\x03" * 4)
            self.assertEqual(src.entries["e.w1.trellis"][0], "model-00001-of-00017.safetensors")


if __name__ == "__main__":
    unittest.main()
