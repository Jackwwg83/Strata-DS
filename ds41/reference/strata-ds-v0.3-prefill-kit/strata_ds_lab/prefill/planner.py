"""Concrete TP1 prefill allocation contract, not a device-memory measurement.
All tensor terms describe the proposed bounded executor. The existing Spark
runtime cannot be presumed to allocate these shapes. A separate measured
workspace/capability gate is required before launching a real model.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from ..common import ContractError, require_int, digest_json
from .layout import Geometry, load_facts, bundle_layout, ring_layout, align, MiB, GiB

@dataclass(frozen=True)
class Inputs:
    ram_gib:int=256
    vram_gib:int=24
    context:int=8192
    slots:int=1
    chunk:int=2048
    ring_mib:int=512
    wave_experts:int=8
    expert_row_tile:int=64
    attention_query_tile:int=64
    attention_key_tile:int=64
    index_query_tile:int=16
    index_key_tile:int=4096
    index_head_tile:int=4
    engram_query_tile:int=256
    kv_format:str='reference_bf16'
    reconstruction:str='disabled'
    mandatory_device_bytes:int|None=None
    external_device_bytes:int=GiB
    device_safety_bytes:int=GiB
    kernel_extra_bytes:int=64*MiB
    sort_extra_bytes:int=32*MiB
    host_system_reserve_bytes:int=18*GiB
    host_runtime_reserve_bytes:int=8*GiB
    host_page_cache_reserve_bytes:int=4*GiB
    engram_row_cache_bytes:int=2*GiB
    host_hot_backup_bytes:int=0
    graph_capture:bool=False
    def __post_init__(self):
        positive=('ram_gib','vram_gib','context','slots','chunk','ring_mib','wave_experts',
                  'expert_row_tile','attention_query_tile','attention_key_tile','index_query_tile',
                  'index_key_tile','index_head_tile','engram_query_tile')
        for name in positive:require_int(getattr(self,name),name,1)
        for name in ('external_device_bytes','device_safety_bytes','kernel_extra_bytes','sort_extra_bytes',
                     'host_system_reserve_bytes','host_runtime_reserve_bytes','host_page_cache_reserve_bytes',
                     'engram_row_cache_bytes','host_hot_backup_bytes'):
            require_int(getattr(self,name),name)
        if self.mandatory_device_bytes is not None:require_int(self.mandatory_device_bytes,'mandatory_device_bytes',1)
        if self.chunk>self.context or self.context>1048576:raise ContractError('chunk/context exceeds model range')
        if self.slots>8:raise ContractError('this initial planning search supports 1..8 slots, not an engine capacity promise')
        if self.wave_experts>384 or self.expert_row_tile>self.chunk:
            raise ContractError('expert wave/row tile exceeds bound')
        if self.index_head_tile>32:raise ContractError('index head tile >32')
        if self.kv_format not in ('reference_bf16','packed_890_experimental'):
            raise ContractError('unknown KV format')
        if self.reconstruction not in ('disabled','one_projection_with_hadamard_temp'):
            raise ContractError('whole-bank/expert reconstruction is not this contract')
        if self.graph_capture is not False:raise ContractError('first contract is eager; graph private pools need another profile')


def state_budget(context:int,slots:int=1,kv_format:str='reference_bf16') -> dict:
    require_int(context,'context',1);require_int(slots,'slots',1)
    if context>1048576:raise ContractError('context exceeds model limit')
    if kv_format=='reference_bf16':main,index=512*2,128*2
    elif kv_format=='packed_890_experimental':main,index=512//2+512//16,128//2+128//32
    else:raise ContractError('unknown cache format')
    # 8 query/reselection sources share FOUR key banks; no additional TP divisor.
    banks=[]
    for layer,ratio in ((2,2),(8,2),(14,2),(20,1)):
        rows=context//ratio
        allocated_rows=align(rows,256)
        banks.append({'source_layer':layer,'compression':ratio,'logical_rows':rows,
                      'allocated_rows':allocated_rows,'bytes':allocated_rows*(main+index)})
    global_bytes=sum(x['bytes'] for x in banks)
    window=40*128*512*2  # conservative all-layer BF16 SWA ownership for this contract
    compressor=3*2*(512+128)*4*2  # ratio2, three producer layers, two FP32 accumulators
    page_tables=sum(x['allocated_rows']//256 for x in banks)*8
    small_metadata=64*1024
    one=global_bytes+window+compressor+page_tables+small_metadata
    return {'format':kv_format,'source_banks':banks,'global_bytes_per_session':global_bytes,
            'swa_bytes_per_session':window,'compressor_bytes_per_session':compressor,
            'page_table_bytes_per_session':page_tables,'metadata_allowance_per_session':small_metadata,
            'bytes_per_session':one,'allocated_slots':slots,'total_bytes':one*slots,
            'asymptotic_global_bytes_per_token':(main+index)*2.5,
            'evidence':'EXPLICIT_PLANNED_LAYOUT; not the legacy v0.1 allocator or stock vLLM allocation'}


def workspace_budget(p:Inputs) -> dict:
    f=load_facts();T=p.chunk;H=f['hidden'];F=f['intermediate'];K=f['top_k']
    Q=min(T,p.attention_query_tile);B=p.attention_key_tile
    IQ=min(T,p.index_query_tile);IK=min(p.context,p.index_key_tile)
    GR=p.wave_experts*min(T,p.expert_row_tile)
    E=min(T,p.engram_query_tile)
    # Simultaneous live tensors, reused phase arena only after last consumer event.
    persistent={
      'hc_residual_pingpong_bf16':2*T*4*H*2,
      'normalized_layer_input_fp16':T*H*2,
      'layer_output_fp32':T*H*4,
      'candidate_block_ids_i32':T*2048*4,
      'selected_key_ids_i32':T*512*4,
      'token_positions_and_flags':T*16,
    }
    moe={
      'contributions_by_token_topk_fp32':T*K*H*4,
      'route_ids_weights_and_stable_sort_arrays':T*K*(4+4+8+8+4+4),
      'counts_offsets':(384+1)*8*3,
      'gate_up_hadamard_inputs_fp16':GR*H*4,
      'gate_up_results_fp32':GR*F*8,
      'down_input_and_hadamard_fp16':GR*F*4,
      'down_result_fp16':GR*H*2,
      'wave_row_metadata':GR*(4+4+4),
      'external_sort_workspace_allowance':p.sort_extra_bytes,
      'native_kernel_extra_allowance':p.kernel_extra_bytes,
    }
    if p.reconstruction!='disabled':
        # Sequential one-matrix FP16 reconstruction plus a separate equal-size
        # Hadamard temporary; this explicitly forbids overlapping reconstructions.
        moe['one_matrix_and_transform_fp16']=2*H*F*2
    attention={
      'query_bf16':Q*64*512*2,
      'kv_gather_shared_across_heads_bf16':Q*B*512*2,
      'score_slab_fp32':Q*64*B*4,
      'online_output_accumulator_fp32':Q*64*512*4,
      'attention_output_bf16':Q*64*512*2,
      'online_max_and_lse_fp32':Q*64*4*2,
      'native_kernel_extra_allowance':p.kernel_extra_bytes,
    }
    indexer={
      'queries_bf16':IQ*32*128*2,
      'key_slab_bf16':IK*128*2,
      'head_score_tile_fp32':IQ*p.index_head_tile*IK*4,
      'accumulated_scores_fp32':IQ*IK*4,
      'topk_merge_pairs_i32_fp32':IQ*(512+IK)*8,
      'candidate_merge_pairs_i32_fp32':IQ*(2048+IK)*8,
      'native_kernel_extra_allowance':p.kernel_extra_bytes,
    }
    engram={
      'raw_fp8_e8m0_rows':E*24*(256+8),
      'decoded_rows_fp32':E*24*256*4,
      'gated_rows_and_projection_fp32':E*(24*256+H)*4,
      'native_kernel_extra_allowance':p.kernel_extra_bytes,
    }
    dense={'gemm_activation_slab_bf16':Q*max(64*512,8*1024)*2,
           'native_kernel_extra_allowance':p.kernel_extra_bytes}
    head={'one_requested_row_logits_fp32':129280*4,
          'head_activation_allowance':H*4*4,
          'native_kernel_extra_allowance':p.kernel_extra_bytes}
    phase_terms={'moe':moe,'attention':attention,'indexer':indexer,'engram':engram,'dense':dense,'head_last_row':head}
    base=sum(persistent.values())
    phase_totals={phase:base+sum(terms.values()) for phase,terms in phase_terms.items()}
    peak_phase=max(phase_totals,key=phase_totals.get)
    return {'persistent':persistent,'phase_terms':phase_terms,'phase_total_bytes':phase_totals,
            'peak_phase':peak_phase,'peak_bytes':phase_totals[peak_phase],
            'row_capacity':T*K,'bounded_live_wave_rows':GR,
            'attention_query_tile_effective':Q,'index_key_tile_effective':IK,
            'head_policy':'last request row only; teacher-forced all-row logits require streamed head or a distinct contract',
            'buffer_reuse_requires':'all last-consumer CUDA events complete; eager only',
            'unknown_runtime_native_requirements':True}


def make_plan(p:Inputs) -> dict:
    f=load_facts();layout=bundle_layout();ring=ring_layout(p.ring_mib*MiB,p.wave_experts)
    ws=workspace_budget(p);state=state_budget(p.context,p.slots,p.kv_format)
    source_residual=f['source_active_payload_bytes']-layout['full_routed_bank_payload_bytes']
    mandatory=p.mandatory_device_bytes if p.mandatory_device_bytes is not None else source_residual
    gpu_terms={'mandatory_device_weight_budget':mandatory,'persistent_session_state':state['total_bytes'],
               'expert_ring':ring['allocated_bytes'],'phase_workspace_peak':ws['peak_bytes'],
               'external_runtime_allowance':p.external_device_bytes,'safety_margin':p.device_safety_bytes}
    gpu_before_hot=sum(gpu_terms.values());total_gpu=p.vram_gib*GiB
    hot_slots=max(0,(total_gpu-gpu_before_hot)//layout['slot_stride_bytes'])
    hot_slots=min(hot_slots,40*384)
    hot_bytes=hot_slots*layout['slot_stride_bytes']
    # Host: two bounded pinned ring arenas for I/O filling and H2D consumers.
    # Immutable CPU weight pool is separate; no full duplicate of GPU-resident bank.
    host_terms={'system_other_and_safety':p.host_system_reserve_bytes,
                'runtime_metadata_and_cpu_scratch':p.host_runtime_reserve_bytes,
                'page_cache_allowance':p.host_page_cache_reserve_bytes,
                'engram_row_cache':p.engram_row_cache_bytes,
                'pinned_expert_staging':2*ring['allocated_bytes'],
                'engram_chunk_double_buffer':2*p.chunk*48*(256+8),
                'gpu_hot_expert_explicit_host_backup':p.host_hot_backup_bytes}
    host_before=sum(host_terms.values());available_host=p.ram_gib*GiB-host_before
    warm_slots=max(0,available_host//layout['slot_stride_bytes'])
    warm_slots=min(warm_slots,40*384-hot_slots)
    warm_bytes=warm_slots*layout['slot_stride_bytes']
    cold_count=40*384-hot_slots-warm_slots
    errors=[]
    if not ring['feasible']:errors.append('ring below two-wave algorithm minimum; never clamp upward')
    if gpu_before_hot>total_gpu:errors.append('mandatory + state + workspace + ring exceeds device design budget')
    if host_before>p.ram_gib*GiB:errors.append('host non-cache terms exceed design budget')
    assumptions=[
       'This TP1 executor is a design contract, not the existing two-Spark runtime.',
       'Full forty-layer prefill; no unqualified CED decoder truncation or speculative acceleration.',
       'Expert admission is route-exact; no top-k pruning; source bundle markers/headers must be audited.',
       'Nonexpert budget defaults to source-ledger active-minus-routed including vision; not a measured allocation.',
       'Native workspace allowances are explicit trial reservations; measured upper bounds must replace them.',
       'KV profile is explicit; packed_890 is an unqualified planned layout, not a claimed vLLM feature.',
       'No whole-model/whole-layer FP16 reconstruction; RAM stores packed expert bytes.',
       'Combined resident capacities assume exclusive placement; they are not a cache hit-rate estimate.',
       'Host page cache is accounted separately; actual cgroup/physical I/O and peak memory still need measurement.']
    out={'schema_version':1,'evidence':'SOURCE_DERIVED_LAYOUT_AND_IMPLEMENTATION_DESIGN',
         'inputs':asdict(p),'model_revision':f['model_revision'],'expert_layout':layout,
         'ring':ring,'workspace':ws,'state':state,
         'device':{'bytes':total_gpu,'terms_before_hot_cache':gpu_terms,
                   'before_hot_cache_bytes':gpu_before_hot,'hot_slots':hot_slots,
                   'hot_cache_bytes':hot_bytes,'planned_peak_bytes':gpu_before_hot+hot_bytes,
                   'remaining_after_safety_bytes':total_gpu-gpu_before_hot-hot_bytes},
         'host':{'bytes':p.ram_gib*GiB,'noncache_terms':host_terms,'warm_slots':warm_slots,
                 'warm_cache_bytes':warm_bytes,'planned_peak_bytes':host_before+warm_bytes,
                 'remaining_after_reserves_bytes':p.ram_gib*GiB-host_before-warm_bytes},
         'cold_expert_count_design':cold_count,'cold_payload_bytes_design':cold_count*layout['source_payload_bytes'],
         'fit_under_declared_design_contract':not errors,'errors':errors,
         'can_launch':False,'performance_prediction':None,
         'launch_blockers':['exact checkpoint audit and checksums','TP1 GPU kernel + model numerical qualification',
                            'measured mandatory/native/allocator peak bounds','actual hardware/cgroup preflight'],
         'assumptions':assumptions}
    out['plan_digest']=digest_json(out)
    return out


def seed_for_recipe(recipe:dict,context:int=8192,slots:int=1,append_tokens:int|None=None)->Inputs:
    if recipe.get('model')!='exl3-3':
        raise ContractError('This exact layout is EXL3 3bpw only; GGUF/SAGE need their own audited bundle layouts')
    v=recipe['vram_gib'];ram=recipe['ram_gib']
    if v not in (16,24):raise ContractError('unreviewed discrete GPU seed')
    chunk=min(context,1024 if v==16 else 2048)
    if append_tokens is not None:
        require_int(append_tokens,'append_tokens',1)
        if append_tokens>context:raise ContractError('append > allocated context')
        chunk=min(chunk,append_tokens)
    return Inputs(ram_gib=ram,vram_gib=v,context=context,slots=slots,chunk=chunk,
                  ring_mib=256 if v==16 else 512,wave_experts=4 if v==16 else 8,
                  expert_row_tile=min(64,chunk))


def candidate_plans(base:Inputs,chunks=(256,512,1024,2048,3072,4096),rings=(256,512,1024)):
    """Enumerate: no monotonic kernel-workspace assumption and no fabricated fastest winner."""
    seen=set();out=[]
    for T in chunks:
        require_int(T,'candidate chunk',1)
        if T>base.context:continue
        for mb in rings:
            p=replace(base,chunk=T,ring_mib=mb,expert_row_tile=min(base.expert_row_tile,T))
            plan=make_plan(p)
            if plan['plan_digest'] not in seen:out.append(plan);seen.add(plan['plan_digest'])
    return out
