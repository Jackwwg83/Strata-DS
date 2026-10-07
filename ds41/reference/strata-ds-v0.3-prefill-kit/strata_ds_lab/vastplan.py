"""Pure offline search/price planning. No credentials, network, create or destroy calls."""
from __future__ import annotations
import math
from .common import ContractError, finite_number, require_int, digest_json

def search_argv(recipe:dict)->list[str]:
    # Vast documents these as MB; use coarse discovery thresholds, NOT an admission proof.
    ram=require_int(recipe['ram_gib'],'ram',1)*1000
    gpu=require_int(recipe['vram_gib'],'vram',1)*1000
    disk=math.ceil(require_int(recipe['storage_free_gib'],'storage',1)*(1<<30)/1e9)
    query=f'verified=true rentable=true rented=false num_gpus=1 cpu_ram>={ram} gpu_ram>={gpu} disk_space>={disk} direct_port_count>=1'
    return ['vastai','search','offers',query,'--raw','-o','dph_total']

def cost_plan(quote:dict,policy:dict)->dict:
    # A normalized quote must explicitly include disk in the hourly number; do not double count it.
    hourly=finite_number(quote['hourly_including_disk_usd'],'hourly')
    hours=finite_number(quote['duration_hours'],'hours',.001)
    down=finite_number(quote['download_gb'],'download GB')
    up=finite_number(quote['upload_gb'],'upload GB')
    down_rate=finite_number(quote['download_usd_per_gb'],'download rate')
    up_rate=finite_number(quote['upload_usd_per_gb'],'upload rate')
    total=hourly*hours+down*down_rate+up*up_rate
    blocks=[]
    for k,value in [('max_hourly_usd',hourly),('max_total_usd',total),('max_duration_hours',hours)]:
        cap=policy.get(k)
        if cap is None:blocks.append(k+' must be approved locally before spending')
        elif value>finite_number(cap,k):blocks.append(k+' exceeded')
    if policy.get('allow_paid_create') is not True:blocks.append('paid creation not authorized')
    if policy.get('max_active_instances')!=1:blocks.append('initial experiments require exactly one active instance')
    return {'status':'OFFLINE_PROPOSAL_ONLY','can_execute':False,
      'quote_sha256':digest_json(quote),'estimated_total_usd':total,'blockers':blocks,
      'required':['Re-query exact offer and price immediately before rental.',
                  'Fresh signed/approved local budget + recipe + image digest + deadline.',
                  'Copy result artifacts before destruction; verify destruction through provider.',
                  'Stopping a job or container does not destroy a rented instance.',
                  'Download charges and disk charges may persist; use actual provider units.']}
