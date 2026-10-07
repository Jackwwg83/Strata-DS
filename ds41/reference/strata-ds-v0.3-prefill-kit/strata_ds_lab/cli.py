from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .common import ContractError, read_json, parse_json
from .planner import plan,traffic_ceiling
from .matrix import make_cases
from .metrics import summarize
from .catalog import audit_files
from .contracts import BACKENDS
from .preflight import probe
from .vastplan import search_argv,cost_plan
ROOT=Path(__file__).resolve().parent.parent

def emit(value):print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def main(argv=None):
    p=argparse.ArgumentParser(description='Offline design tools: no GPU inference, no cloud mutations')
    p.add_argument('--root',type=Path,default=ROOT,help='checkout containing specs/ and recipes/')
    sp=p.add_subparsers(dest='command',required=True)
    x=sp.add_parser('plan');x.add_argument('recipe',nargs='?');x.add_argument('--all',action='store_true')
    x=sp.add_parser('matrix');x.add_argument('--stage',choices=['smoke','screen','full'],default='smoke');x.add_argument('--recipes',nargs='*')
    x=sp.add_parser('audit-safetensors');x.add_argument('checkpoint',type=Path);x.add_argument('files',nargs='+')
    sp.add_parser('backend-status')
    x=sp.add_parser('probe');x.add_argument('--path',type=Path,default=Path('.'));x.add_argument('--cgroup',type=Path)
    x=sp.add_parser('summarize');x.add_argument('jsonl',type=Path)
    x=sp.add_parser('vast-search');x.add_argument('recipe')
    x=sp.add_parser('cost-plan');x.add_argument('quote');x.add_argument('--policy',default='specs/cloud-policy.json')
    x=sp.add_parser('traffic');x.add_argument('--expert-bytes',type=int,required=True);x.add_argument('--gpu-hit',type=float,required=True)
    x.add_argument('--ram-hit-given-gpu-miss',type=float,required=True);x.add_argument('--ssd-Bps',type=float,required=True);x.add_argument('--h2d-Bps',type=float,required=True)
    a=p.parse_args(argv)
    def recipe(rid):
        # Never interpret a recipe ID as an arbitrary path.
        if not rid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-.' for c in rid):raise ContractError('invalid recipe ID')
        return read_json(a.root/'recipes'/f'{rid}.json')
    try:
      if a.command=='plan':
        models=read_json(a.root/'specs/models.json')['models']
        if a.all:
            rows=[plan(r,models[r['model']]) for r in [read_json(f) for f in sorted((a.root/'recipes').glob('*.json'))]];emit(rows)
        elif a.recipe:
            r=recipe(a.recipe);emit(plan(r,models[r['model']]))
        else:raise ContractError('supply a recipe ID or --all')
      elif a.command=='matrix':
        rs=[recipe(rid) for rid in a.recipes] if a.recipes else [read_json(f) for f in sorted((a.root/'recipes').glob('*.json'))]
        if not a.recipes:rs=[r for r in rs if r['tier']=='core']
        for row in make_cases(rs,a.stage):print(json.dumps(row,ensure_ascii=False,allow_nan=False))
      elif a.command=='audit-safetensors':emit(audit_files(a.checkpoint,a.files))
      elif a.command=='backend-status':emit(BACKENDS)
      elif a.command=='probe':emit(probe(a.path,a.cgroup))
      elif a.command=='summarize':
        if a.jsonl.stat().st_size>64<<20:raise ContractError('timing file exceeds cap; partition by trial')
        rows=[parse_json(line) for line in a.jsonl.read_text().splitlines() if line.strip()];emit(summarize(rows))
      elif a.command=='vast-search':
        emit({'argv':search_argv(recipe(a.recipe)),'executed':False,
              'note':'Only prints read-only discovery command; advertised MB thresholds do not certify allocated GiB.'})
      elif a.command=='cost-plan':emit(cost_plan(read_json(a.quote),read_json(a.root/a.policy)))
      elif a.command=='traffic':emit(traffic_ceiling(a.expert_bytes,a.gpu_hit,a.ram_hit_given_gpu_miss,a.ssd_Bps,a.h2d_Bps))
      return 0
    except (ContractError,OSError,KeyError,TypeError,ValueError) as e:
      print(f'ERROR: {e}',file=sys.stderr);return 2
