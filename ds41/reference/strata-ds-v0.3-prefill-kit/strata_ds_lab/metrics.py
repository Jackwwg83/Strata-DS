"""Strict aggregation of engine-level timing records, not SSE chunks as tokens."""
from __future__ import annotations
from .common import ContractError, require_int

def percentile(values:list[float],q:float):
    if not 0<=q<=1: raise ContractError('percentile q out of range')
    if not values:return None
    v=sorted(values);i=(len(v)-1)*q;a=int(i);b=min(a+1,len(v)-1)
    return v[a]+(v[b]-v[a])*(i-a)

def summarize(records:list[dict])->dict:
    if not records:raise ContractError('empty trial')
    # One trial / clock origin / immutable condition only. Do not pool runs here.
    identity_keys=('trial_id','clock_id','condition_sha256','provenance')
    identity=tuple(records[0].get(k) for k in identity_keys)
    if any(not isinstance(v,str) or not v for v in identity):raise ContractError('missing trial identity')
    if identity[-1] not in {'synthetic_test','measured_engine'}:raise ContractError('invalid provenance')
    ids=set();ttft=[];tpot=[];per_request=[];all_tokens=0;good_tokens=0;completed=0
    starts=[];ends=[]
    for r in records:
        if tuple(r.get(k) for k in identity_keys)!=identity:raise ContractError('mixed trial/clock/condition/provenance')
        rid=r.get('request_id')
        if not isinstance(rid,str) or not rid or rid in ids:raise ContractError('duplicate/missing request ID')
        ids.add(rid)
        if r.get('timing_source')!='engine_token':raise ContractError('SSE chunks are not token timings')
        s=require_int(r['submit_ns'],'submit_ns');end=require_int(r['end_ns'],'end_ns')
        ts=r['token_timestamps_ns']
        if not isinstance(ts,list):raise ContractError('token timestamps must be list')
        for t in ts:require_int(t,'token timestamp')
        if end<s or ts!=sorted(ts) or any(t<s or t>end for t in ts):raise ContractError('invalid clock ordering')
        if require_int(r['output_tokens'],'output_tokens')!=len(ts):raise ContractError('token count mismatch')
        status=r.get('status')
        if status not in {'completed','cancelled','error','timeout'}:raise ContractError('unknown status')
        starts.append(s);ends.append(end);all_tokens+=len(ts)
        if status=='completed':
            if not ts:raise ContractError('completed benchmark request must produce tokens')
            completed+=1;good_tokens+=len(ts);ttft.append((ts[0]-s)/1e6)
            tpot.extend((b-a)/1e6 for a,b in zip(ts,ts[1:]))
            per_request.append({'request_id':rid,'ttft_ms':(ts[0]-s)/1e6,
                 'decode_tokens_per_second':(len(ts)-1)*1e9/(ts[-1]-ts[0]) if len(ts)>1 and ts[-1]>ts[0] else None})
    wall=(max(ends)-min(starts))/1e9
    if wall<=0:raise ContractError('zero trial elapsed time')
    return {'trial_id':identity[0],'condition_sha256':identity[2],'provenance':identity[3],
      'requests':len(records),'completed_requests':completed,'failure_or_cancel_rate':1-completed/len(records),
      'wall_seconds':wall,'goodput_output_tokens':good_tokens,'all_output_tokens':all_tokens,
      'aggregate_goodput_tok_s':good_tokens/wall,'per_completed_request':per_request,
      'ttft_ms':{k:percentile(ttft,q) for k,q in [('p50',.5),('p95',.95),('p99',.99)]},
      'tpot_ms':{k:percentile(tpot,q) for k,q in [('p50',.5),('p95',.95),('p99',.99)]},
      'note':'Goodput includes trial wall time of failed/cancelled requests; per-request rates are never summed.'}
