from __future__ import annotations
from .common import ContractError, digest_json, require_int

def make_cases(recipes: list[dict],stage:str='smoke')->list[dict]:
    if stage not in {'smoke','screen','full'}:raise ContractError('unknown stage')
    cases=[];seen=set()
    for recipe in recipes:
        if recipe['id'] in seen:raise ContractError('duplicate recipe')
        seen.add(recipe['id'])
        contexts=[1024] if stage=='smoke' else ([1024,8192,32768] if stage=='screen' else recipe['context_steps'])
        concurrencies=[1] if stage!='full' else recipe['concurrency_steps']
        conditions=['model_cold'] if stage=='smoke' else ['warm_model_new_prefix','warm_agent_append']
        repeats=1 if stage=='smoke' else 3
        generated=16 if stage=='smoke' else 256
        for ctx in contexts:
          require_int(ctx,'context',generated+1)
          for c in concurrencies:
            for condition in conditions:
              require_int(c,'concurrency',1)
              for rep in range(repeats):
                case={'recipe_id':recipe['id'],'recipe_sha256':digest_json(recipe),'stage':stage,
                  'context_allocated_tokens':ctx,'prompt_target_tokens':ctx-generated,
                  'output_token_limit':generated,'ready_sessions':c,'cache_condition':condition,
                  'repeat':rep,'speculation':False,'vision':False,
                  'required_gates':['artifact_integrity', 'operator_parity' if stage=='smoke' else 'same_quant_offload_parity','hardware_preflight'],
                  'append_tokens':min(256,ctx-generated) if condition=='warm_agent_append' else 0,
                  'prior_prefix_tokens':max(0,ctx-generated-256) if condition=='warm_agent_append' else 0,
                  'prompt_materialization':'must_use_pinned_tokenizer_and_template; not character estimates',
                  'status':'PLANNED_NOT_EXECUTED'}
                case['case_id']=digest_json(case)[:20];cases.append(case)
    return cases
