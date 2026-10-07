from __future__ import annotations
from .common import ContractError, GiB, require_int, finite_number, digest_json

def plan(recipe: dict, model: dict) -> dict:
    if require_int(recipe.get('schema_version'), 'schema_version', 1) != 1 or recipe.get('topology') != 'discrete_single_gpu':
        raise ContractError('only schema 1 discrete_single_gpu is modeled')
    ram = require_int(recipe['ram_gib'], 'ram_gib', 1) * GiB
    vram = require_int(recipe['vram_gib'], 'vram_gib', 1) * GiB
    reserves = recipe['host_reserve_gib']
    required = {'os_other_and_margin','engram_rows','io_staging','runtime_kv_metadata'}
    if set(reserves) != required:
        raise ContractError('all host reserve components must be explicit')
    reserve = sum(require_int(v, k) * GiB for k,v in reserves.items())
    dreserve = require_int(recipe['device_nonweight_reserve_gib'], 'device reserve') * GiB
    if reserve >= ram or dreserve >= vram:
        raise ContractError('no positive weight capacity remains')
    if recipe['engram_placement'] != 'ssd_native_fp8_lossless':
        raise ContractError('this planner requires explicitly SSD-backed native Engram')
    weight = require_int(model['non_engram_hint_bytes'], 'non_engram_hint_bytes', 1)
    host_weight = ram - reserve
    gpu_weight = vram - dreserve
    optimistic_unique = host_weight + gpu_weight
    spill = max(0, weight - optimistic_unique)
    return {
      'status':'DESIGN_ESTIMATE', 'can_launch':False, 'recipe_id':recipe['id'],
      'recipe_sha256':digest_json(recipe), 'model_revision':model['revision'],
      'host_weight_ceiling_bytes':host_weight,
      'gpu_all_weights_ceiling_bytes':gpu_weight,
      'optimistic_unique_weight_ceiling_bytes':optimistic_unique,
      'non_engram_hint_bytes':weight,
      'cold_weight_lower_bound_bytes':spill,
      'cold_weight_lower_bound_gib':spill / GiB,
      'possible_all_active_weight_residency': None if 'LOWER_BOUND' in model['size_evidence'] else spill == 0,
      'size_evidence':model['size_evidence'],
      'engram_ssd_hint_bytes':model['engram_hint_bytes'],
      'mandatory_gpu_bytes':None, 'expert_gpu_cache_bytes':None,
      'blockers':['backend integration and hardware qualification missing',
                  'exact text/vision/draft tensor partition required',
                  'mandatory GPU working set, decode/prefill peaks and cgroup must be measured',
                  'zero spill does not prove fit; capacity share is not request hit rate'],
      'caveats':['CPU/GPU sums are optimistic unique-weight bounds, not a shared allocator.',
                 'No extra full CPU copy of GPU weights is budgeted; staging must bound temporary duplicates.',
                 'Page cache is charged memory; buffered I/O prototypes do not enforce this RAM plan.',
                 'Context/concurrency reserves must be replaced with backend-specific measurements.']}

def traffic_ceiling(expert_bytes_per_token, gpu_request_byte_hit_rate,
                    ram_given_gpu_miss_byte_hit_rate, ssd_Bps, h2d_Bps,
                    execution='gpu_stream_packed'):
    e=finite_number(expert_bytes_per_token,'expert bytes',1)
    hg=finite_number(gpu_request_byte_hit_rate,'gpu hit')
    hr=finite_number(ram_given_gpu_miss_byte_hit_rate,'RAM conditional hit')
    s=finite_number(ssd_Bps,'SSD B/s',1)
    p=finite_number(h2d_Bps,'H2D B/s',1)
    if hg>1 or hr>1: raise ContractError('hit rates must be <= 1')
    if execution!='gpu_stream_packed':
        raise ContractError('CPU expert compute needs a separate calibrated model')
    h2d=e*(1-hg)
    ssd=h2d*(1-hr)
    floor=max(ssd/s,h2d/p)
    return {'label':'OPTIMISTIC_IO_ONLY_CEILING_NOT_THROUGHPUT_PREDICTION',
      'ssd_logical_bytes_per_token':ssd,'h2d_packed_bytes_per_token':h2d,
      'overlapped_transfer_seconds_lower_bound':floor,
      'tokens_per_second_upper_bound':1/floor if floor else None,
      'excluded':'compute, SSD read amplification, layer dependency and launch latency, activation transfers, Engram, speculation'}
