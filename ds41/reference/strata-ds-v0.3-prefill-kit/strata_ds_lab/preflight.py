"""Read-only host evidence collector. It does not apply cgroups, change swap or drop caches."""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
from pathlib import Path
from .common import ContractError

def parse_meminfo(text:str)->dict:
    out={}
    for line in text.splitlines():
        k,sep,rest=line.partition(':')
        if not sep:continue
        parts=rest.split()
        if not parts:continue
        if len(parts)>1 and parts[1]!='kB':raise ContractError('unexpected meminfo unit')
        out[k]=int(parts[0])*1024
    return out

def inspect_cgroup_chain(current:Path,mount:Path)->dict:
    current=current.resolve();mount=mount.resolve()
    if current!=mount and mount not in current.parents:raise ContractError('cgroup leaves mount')
    evidence=[];remaining=[];missing=[]
    for p in [current,*current.parents]:
        if p!=mount and mount not in p.parents:break
        row={'relative':str(p.relative_to(mount))}
        for name in ('memory.max','memory.current','memory.swap.max','memory.swap.current','memory.events','memory.stat','io.stat','cpuset.cpus.effective'):
            try:row[name]=(p/name).read_text().strip()
            except OSError:row[name]=None
        if row['memory.max'] is None or row['memory.current'] is None:
            missing.append(str(p))
        elif row['memory.max']!='max':
            remaining.append(max(0,int(row['memory.max'])-int(row['memory.current'])))
        evidence.append(row)
        if p==mount:break
    return {'ancestors':evidence,'remaining_bytes':min(remaining) if remaining else None,
            'complete':not missing,'missing_paths':missing}

def probe(path:Path=Path('.'),explicit_cgroup:Path|None=None)->dict:
    out={'status':'OBSERVATION_NOT_QUALIFICATION','platform':platform.platform(),'machine':platform.machine(),
      'logical_cpu_count':os.cpu_count(),'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
      'disk':dict(zip(('total','used','free'),shutil.disk_usage(path))),
      'cgroup':None,'cgroup_discovery_complete':False}
    try:out['meminfo']=parse_meminfo(Path('/proc/meminfo').read_text())
    except OSError:out['meminfo']=None
    try:out['self_cgroup']=Path('/proc/self/cgroup').read_text()
    except OSError:out['self_cgroup']=None
    try:out['mountinfo']=Path('/proc/self/mountinfo').read_text()
    except OSError:out['mountinfo']=None
    mount=Path('/sys/fs/cgroup')
    cg=explicit_cgroup
    if cg is None and out['self_cgroup']:
        for line in out['self_cgroup'].splitlines():
            if line.startswith('0::'):
                relative=line[3:].lstrip('/')
                candidate=mount/relative
                if '..' not in Path(relative).parts and candidate.exists():cg=candidate
    if cg is not None:
        try:
            out['cgroup']=inspect_cgroup_chain(cg,mount)
            out['cgroup_discovery_complete']=out['cgroup']['complete']
        except (OSError,ValueError) as e:out['cgroup_error']=str(e)
    gpu=shutil.which('nvidia-smi')
    if gpu:
        try:
            p=subprocess.run([gpu,'--query-gpu=name,uuid,memory.total,memory.free,driver_version','--format=csv,noheader'],
                 capture_output=True,text=True,timeout=10,check=False)
            out['nvidia_smi']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except (OSError,subprocess.TimeoutExpired) as e:out['nvidia_smi']={'error':str(e)}
    else:out['nvidia_smi']=None
    out['required_before_real_run']=['Verify cgroup ancestors, allocation, cpuset and actual CUDA free bytes.',
        'Validate GPU kernels and independently measure pinned H2D/CPU bandwidth.',
        'Measure NVMe read workload; advertised disk_bw is not a measurement.',
        'A large-host budget-cap run is not a physical small-host result.']
    return out
