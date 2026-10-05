"""Validate the complete design kit; never imports CUDA or creates cloud resources."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from strata_ds_lab.common import read_json,ContractError,safe_path
from strata_ds_lab.prefill.layout import ROOT,bundle_layout,load_facts
from strata_ds_lab.prefill.planner import seed_for_recipe,make_plan

def main():
    load_facts()
    layout=bundle_layout()
    if (layout['source_payload_bytes'],layout['slot_stride_bytes'])!=(13315596,13316352):
        raise ContractError('target EXL3 byte contract changed')
    preserved=read_json(ROOT/'specs/prefill/preserved-files.json')
    for item in preserved['files']:
        b=safe_path(ROOT,item['path']).read_bytes()
        if hashlib.sha256(b).hexdigest()!=item['sha256']:raise ContractError('preserved recipe/model pin changed')
    backlog=read_json(ROOT/'specs/backlog.json')
    u39=read_json(ROOT/'specs/upstream-v0139.backlog.json')
    new=read_json(ROOT/'specs/prefill/backlog.json')
    tasks=backlog['tasks']+u39['tasks']+new['tasks']
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)):raise ContractError('duplicate task ID')
    deps={t['id']:set(t.get('depends_on',[])) for t in tasks}
    # The original U39 validator independently validates its augmentation format.
    for i,d in deps.items():
        if d-set(ids):raise ContractError(f'missing dependency for {i}: {d-set(ids)}')
    done=[]
    while len(done)<len(ids):
        candidates=sorted(i for i,d in deps.items() if i not in done and d<=set(done))
        if not candidates:raise ContractError('dependency cycle')
        done.extend(candidates)
    count=0
    for p in (ROOT/'recipes').glob('*.json'):
        recipe=read_json(p)
        if recipe['model']=='exl3-3':
            plan=make_plan(seed_for_recipe(recipe));count+=1
            if plan['can_launch'] is not False:raise ContractError('offline plan mislabeled real launch')
    print(json.dumps({'status':'CONTRACTS_AND_PRESERVED_FILES_VALID_NOT_GPU_QUALIFICATION',
                      'exl3_seed_recipes':count,'tasks':len(tasks),'topological_order':done,
                      'can_launch':False},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
