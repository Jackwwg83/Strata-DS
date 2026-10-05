"""Original schema arithmetic from pinned public V4.1/MUL1 interfaces.
No upstream kernel code is copied. Source-derived sizes != audited checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from ..common import ContractError, require_int, read_json

ROOT = Path(__file__).resolve().parents[2]
FACTS_PATH = ROOT / 'specs/prefill/deepseek-v41.facts.json'
MUL1 = 0x83DCD12D
MiB = 1 << 20
GiB = 1 << 30

def align(n: int, quantum: int = 256) -> int:
    require_int(n, 'bytes'); require_int(quantum, 'alignment', 1)
    if quantum & (quantum - 1): raise ContractError('alignment must be a power of two')
    return (n + quantum - 1) // quantum * quantum

@dataclass(frozen=True)
class Geometry:
    hidden: int = 5120
    intermediate: int = 2304
    layers: int = 40
    experts: int = 384
    top_k: int = 6
    bits: int = 3
    def __post_init__(self):
        for name, val in asdict(self).items(): require_int(val, name, 1)
        if self.hidden % 128 or self.intermediate % 128:
            raise ContractError('MUL1 dimensions must align to 128-channel Hadamard blocks')
        if self.top_k > self.experts: raise ContractError('top_k exceeds expert count')
        if self.bits not in (3, 4): raise ContractError('only the audited MUL1 3/4-bit schema is supported')

@dataclass(frozen=True)
class Component:
    projection: str
    suffix: str
    dtype: str
    shape: tuple[int, ...]
    payload_bytes: int
    slot_offset: int


def components(g: Geometry = Geometry(), alignment: int = 256) -> tuple[Component, ...]:
    """TP1 bundle. Every component start is aligned; file offsets are NOT inferred."""
    offset = 0; out = []
    for projection in ('w1', 'w3', 'w2'):
        i, o = (g.intermediate, g.hidden) if projection == 'w2' else (g.hidden, g.intermediate)
        specs = [('trellis', 'I16', (i//16, o//16, g.bits*16), i*o*g.bits//8),
                 ('suh', 'F16', (i,), i*2), ('svh', 'F16', (o,), o*2),
                 ('mul1', 'I32', (1,), 4)]
        for suffix, dtype, shape, size in specs:
            offset = align(offset, alignment)
            out.append(Component(projection, suffix, dtype, shape, size, offset))
            offset += size
    return tuple(out)


def bundle_layout(g: Geometry = Geometry(), alignment: int = 256) -> dict:
    c = components(g, alignment)
    stride = align(c[-1].slot_offset + c[-1].payload_bytes, alignment)
    payload = sum(x.payload_bytes for x in c)
    return dict(geometry=asdict(g), topology='TP1', source_payload_bytes=payload,
                slot_stride_bytes=stride, component_alignment=alignment,
                slot_padding_bytes=stride-payload,
                full_layer_payload_bytes=payload*g.experts,
                full_routed_bank_payload_bytes=payload*g.experts*g.layers,
                one_fp16_reconstructed_matrix_bytes=g.hidden*g.intermediate*2,
                components=[asdict(x) for x in c], evidence='PINNED_SOURCE_SCHEMA_DERIVATION')


def load_facts() -> dict:
    f=read_json(FACTS_PATH)
    if (f['hidden'],f['intermediate'],f['layers'],f['experts'],f['top_k']) != (5120,2304,40,384,6):
        raise ContractError('unreviewed model geometry')
    if f['layer_bits'] != [3]*40 or f['codebook'] != 'mul1':
        raise ContractError('target is full-pool all-3-bit MUL1, not SAGE/3.25/GGUF')
    if f['compress_ratios'] != [0,0]+[2]*18+[1]*20:
        raise ContractError('unreviewed compression layout')
    return f


def validate_target_config(config: dict) -> dict:
    """Reject incompatible HF text geometry; do not infer format solely from its name."""
    t=config.get('text_config',config)
    expected={'hidden_size':5120,'moe_intermediate_size':2304,'num_hidden_layers':40,
              'n_routed_experts':384,'num_experts_per_tok':6,'n_shared_experts':1,
              'num_attention_heads':64,'head_dim':512,'index_head_dim':128,
              'index_n_heads':32,'sliding_window':128,'hc_mult':4}
    for k,v in expected.items():
        if type(t.get(k)) is not int or t[k]!=v: raise ContractError(f'incompatible/missing config field {k}')
    for k,v in {'kv_source_layer_ids':[2,8,14,20],
                'index_source_layer_ids':[2,8,14,20,24,28,32,36],
                'engram_layer_ids':[1,14]}.items():
        if t.get(k)!=v: raise ContractError(f'incompatible {k}')
    ratios=t.get('compress_ratios')
    if not isinstance(ratios,list) or ratios[:40]!=[0,0]+[2]*18+[1]*20:
        raise ContractError('incompatible compression ratios')
    if len(ratios) not in (40,43) or (len(ratios)==43 and ratios[40:]!=[0,0,0]):
        raise ContractError('unreviewed draft suffix')
    return {'status':'TEXT_GEOMETRY_MATCHED_NOT_WEIGHT_FORMAT_OR_QUALITY_VERIFIED'}


def ring_layout(budget_bytes: int, wave_experts: int, g: Geometry = Geometry()) -> dict:
    require_int(budget_bytes,'ring_budget');require_int(wave_experts,'wave_experts',1)
    if wave_experts>g.experts: raise ContractError('wave exceeds a layer')
    stride=bundle_layout(g)['slot_stride_bytes']
    slots=budget_bytes//stride
    need=2*wave_experts  # this design explicitly overlaps two whole-expert waves
    return {'budget_bytes':budget_bytes,'slot_stride_bytes':stride,'slots':slots,
            'allocated_bytes':slots*stride,'unused_bytes':budget_bytes-slots*stride,
            'wave_experts':wave_experts,'min_slots':need,'feasible':slots>=need,
            'algorithm':'two_wave_whole_expert_lease','bytes_are_device_only':True}
