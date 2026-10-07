"""Deterministic, bounded wave schedule for known layer routes.
The CPU reference does not simulate bandwidth or implement EXL3 arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
from ..common import ContractError, require_int, finite_number
from .layout import Geometry, bundle_layout

@dataclass(frozen=True)
class Assignment:
    token: int
    slot: int
    weight: float


def group_routes(ids: list[list[int]], weights: list[list[float]], g: Geometry=Geometry()):
    if len(ids)!=len(weights):raise ContractError('routing row mismatch')
    groups={}
    for token,(row,wrow) in enumerate(zip(ids,weights)):
        if len(row)!=g.top_k or len(wrow)!=g.top_k:raise ContractError('routing width must equal model top_k')
        seen=set()
        for slot,(expert,w) in enumerate(zip(row,wrow)):
            require_int(expert,'expert')
            if expert>=g.experts or expert in seen:raise ContractError('invalid/duplicate expert in token top_k')
            seen.add(expert);w=finite_number(w,'routing weight')
            groups.setdefault(expert,[]).append(Assignment(token,slot,w))
    return dict(sorted(groups.items()))


def make_waves(groups: dict, wave_experts: int, row_tile: int, *, gpu_ready=(),ram_ready=(),
               g:Geometry=Geometry()) -> dict:
    require_int(wave_experts,'wave_experts',1);require_int(row_tile,'row_tile',1)
    if wave_experts>g.experts:raise ContractError('wave too large')
    gpu=set(gpu_ready);ram=set(ram_ready)
    for e in gpu|ram:
        require_int(e,'resident expert')
        if e>=g.experts:raise ContractError('resident ID outside layer')
    payload=bundle_layout(g)['source_payload_bytes'];waves=[]
    stats={'requested_unique_experts':len(groups),'requested_bundle_bytes':len(groups)*payload,
           'gpu_hit_bytes':0,'ram_hit_on_gpu_miss_bytes':0,'ssd_logical_bundle_bytes':0,'h2d_payload_bytes':0,
           'expert_loads':0,'assignments':0,'segments':0}
    items=list(groups.items())
    for start in range(0,len(items),wave_experts):
        jobs=[]
        for e,rows in items[start:start+wave_experts]:
            if type(e) is not int or not 0<=e<g.experts or not rows:raise ContractError('invalid group')
            if e in gpu:source='gpu';stats['gpu_hit_bytes']+=payload
            else:
                source='ram' if e in ram else 'ssd'
                stats['h2d_payload_bytes']+=payload;stats['expert_loads']+=1
                stats['ram_hit_on_gpu_miss_bytes' if source=='ram' else 'ssd_logical_bundle_bytes']+=payload
            segments=[rows[i:i+row_tile] for i in range(0,len(rows),row_tile)]
            jobs.append({'expert':e,'source':source,'row_count':len(rows),'segments':segments,
                         'lease_until':'all_segments_and_last_device_event'})
            stats['assignments']+=len(rows);stats['segments']+=len(segments)
        waves.append(jobs)
    return {'waves':waves,'stats':stats,'physical_io_bytes':None,
            'all_experts_streamed':False,'ordering':'expert_id_ascending; token_then_topk_slot_within_expert',
            'expert_loaded_once_per_chunk_layer':True}


def execute_reference(ids,weights,provider:Callable,compute:Callable,*,wave_experts=4,row_tile=64,
                      g:Geometry=Geometry(),output_width:int=1):
    """Executable CPU scheduler. compute gets the weight BEFORE down projection.
    provider(expert) is a context manager. A whole expert is leased through all
    row tiles; contributions are stored by original top-k slot then combined in
    that slot order. No token/route weights are dropped on missing experts.
    GPU integration must replace the synchronous exit by real device fences.
    """
    require_int(output_width,'output_width',1)
    groups=group_routes(ids,weights,g);plan=make_waves(groups,wave_experts,row_tile,g=g)
    contributions=[[None]*g.top_k for _ in ids]
    from contextlib import ExitStack
    for wave in plan['waves']:
        with ExitStack() as stack:
            handles={job['expert']:stack.enter_context(provider(job['expert'])) for job in wave}
            for job in wave:
                handle=handles[job['expert']]
                if handle is None:raise ContractError('missing expert: TP1 must not skip it')
                for segment in job['segments']:
                    values=compute(handle,segment)
                    if len(values)!=len(segment):raise ContractError('kernel returned wrong number of rows')
                    for a,value in zip(segment,values):
                        if len(value)!=output_width or any(not math.isfinite(float(v)) for v in value):
                            raise ContractError('kernel output shape/nonfinite error')
                        if contributions[a.token][a.slot] is not None:raise ContractError('duplicate assignment')
                        contributions[a.token][a.slot]=list(value)
    out=[]
    for row in contributions:
        if any(v is None for v in row):raise ContractError('uncomputed contribution')
        # Fixed ordinary addition order, not a claim of FP32/native bit-exactness.
        result=[0.0]*output_width
        for value in row:
            for j,v in enumerate(value):result[j]+=v
        out.append(result)
    return out,plan['stats']
