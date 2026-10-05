"""Choose ONLY from measured, identity-matched candidate records.
A feasible arithmetic plan is never converted to a speed claim.
"""
from __future__ import annotations
import statistics
from ..common import ContractError, finite_number, require_int
IDENTITY=('model_revision','quant_digest','engine_commit','hardware_digest',
          'workload_digest','kv_format','configured_slots','math_policy','warmth_policy')

def choose_measured(records:list[dict],*,max_decode_gap_ms:float,minimum_repeats:int=3) -> dict:
    finite_number(max_decode_gap_ms,'max_decode_gap_ms');require_int(minimum_repeats,'minimum_repeats',3)
    if not records:raise ContractError('no measurements')
    identity=None;groups={};trials=set()
    for r in records:
        if r.get('evidence')!='MODEL_PERFORMANCE_MEASURED' or r.get('correctness_passed') is not True:
            raise ContractError('synthetic/component/unqualified timing cannot select a model deployment')
        if r.get('peak_within_budget') is not True:raise ContractError('memory bound not verified')
        key=tuple(r.get(k) for k in IDENTITY)
        if any(x is None or x=='' for x in key):raise ContractError('missing comparison identity')
        if identity is None:identity=key
        elif key!=identity:raise ContractError('unmatched hardware/model/workload/math identity')
        trial=r.get('trial_id');candidate=r.get('candidate_id')
        if not isinstance(trial,str) or not trial or trial in trials:raise ContractError('missing/duplicate trial ID')
        trials.add(trial)
        if not isinstance(candidate,str) or not candidate:raise ContractError('candidate_id required')
        required=('prefill_ms','refill_ms','following_decode_ms','max_decode_gap_ms')
        values={k:finite_number(r.get(k),k) for k in required}
        for k in ('ssd_physical_bytes','h2d_payload_bytes','gpu_peak_bytes','host_peak_bytes'):
            require_int(r.get(k),k)
        groups.setdefault(candidate,[]).append(values)
    scores=[]
    for candidate,rows in groups.items():
        if len(rows)<minimum_repeats:raise ContractError('each candidate needs independent repeats')
        times=[r['prefill_ms']+r['refill_ms']+r['following_decode_ms'] for r in rows]
        valid=max(r['max_decode_gap_ms'] for r in rows)<=max_decode_gap_ms
        scores.append({'candidate_id':candidate,'repeats':len(rows),'median_objective_ms':statistics.median(times),
                       'minimum_ms':min(times),'maximum_ms':max(times),'qos_passed':valid})
    good=[s for s in scores if s['qos_passed']]
    return {'status':'MEASURED_CANDIDATE_SELECTED' if good else 'NO_CANDIDATE_PASSES_QOS',
            'winner':min(good,key=lambda s:(s['median_objective_ms'],s['candidate_id']))['candidate_id'] if good else None,
            'scores':scores,'automatic_cloud_or_engine_changes':False,
            'objective':'prefill + cache-refill + fixed following decode workload; not prefill tokens/s alone'}
