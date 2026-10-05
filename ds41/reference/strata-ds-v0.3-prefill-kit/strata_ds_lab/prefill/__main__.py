from __future__ import annotations
import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from ..common import ContractError, read_json, digest_json
from .layout import load_facts,bundle_layout,ROOT,validate_target_config
from .planner import make_plan,seed_for_recipe,candidate_plans
from .audit import audit_exl3
from .tune import choose_measured

def main(argv=None):
    parser=argparse.ArgumentParser(description='DeepSeek V4.1 exact-schema prefill design tools; no model server/cloud writes')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('facts')
    for name in ('plan','matrix'):
        p=sub.add_parser(name);p.add_argument('--recipe',default='exl3-3-r256-v24')
        p.add_argument('--context',type=int,default=8192);p.add_argument('--slots',type=int,default=1)
        p.add_argument('--append-tokens',type=int);p.add_argument('--chunk',type=int)
        p.add_argument('--ring-mib',type=int);p.add_argument('--wave-experts',type=int)
        p.add_argument('--kv-format',choices=['reference_bf16','packed_890_experimental'],default='reference_bf16')
    p=sub.add_parser('audit');p.add_argument('--checkpoint',required=True)
    p.add_argument('--files-list',required=True,help='JSON array of relative shard names; explicit bounded set')
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--config',help='actual target Hugging Face config.json; required for full coverage audit')
    p=sub.add_parser('select');p.add_argument('--records',required=True);p.add_argument('--max-decode-gap-ms',type=float,required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='facts':out={'facts':load_facts(),'layout':bundle_layout()}
        elif args.action in ('plan','matrix'):
            if '/' in args.recipe or '\\' in args.recipe or '..' in args.recipe:raise ContractError('invalid recipe name')
            recipe=read_json(ROOT/'recipes'/f'{args.recipe}.json')
            p=seed_for_recipe(recipe,args.context,args.slots,args.append_tokens)
            changes={'kv_format':args.kv_format}
            for name in ('chunk','ring_mib','wave_experts'):
                if getattr(args,name) is not None:changes[name]=getattr(args,name)
            if args.chunk is not None:changes['expert_row_tile']=min(p.expert_row_tile,args.chunk)
            p=replace(p,**changes)
            if args.action=='plan':out=make_plan(p)
            else:
                plans=candidate_plans(p)
                out={'recipe':args.recipe,'candidate_count':len(plans),'automatic_winner':None,
                     'plans':plans,'note':'enumerated design fits, not timings or launch authorization'}
        elif args.action=='audit':
            names=read_json(args.files_list)
            if not isinstance(names,list) or not all(isinstance(x,str) for x in names):raise ContractError('files-list must be a JSON string array')
            if args.require_complete and not args.config:raise ContractError('full model audit requires the actual target --config')
            config=read_json(args.config) if args.config else None
            geometry_check=validate_target_config(config) if config is not None else {'status':'TEXT_CONFIG_NOT_CHECKED'}
            out=audit_exl3(args.checkpoint,names,complete=args.require_complete)
            out['text_geometry_check']=geometry_check
            out['config_semantic_digest']=digest_json(config) if config is not None else None
        else:out=choose_measured(read_json(args.records),max_decode_gap_ms=args.max_decode_gap_ms)
        print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False));return 0
    except (ContractError,ValueError,OSError,KeyError) as e:
        print(json.dumps({'error':str(e),'can_launch':False},ensure_ascii=False),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
