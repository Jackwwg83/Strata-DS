"""Verify a distribution's file hashes; integrity is not publisher authenticity."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from strata_ds_lab.common import read_json,safe_path,ContractError

def main():
    manifest=read_json(root/'release-manifest.json')
    seen=set()
    for item in manifest['files']:
        name=item['path']
        if name in seen:raise ContractError('duplicate manifest path')
        seen.add(name);p=safe_path(root,name)
        if p.stat().st_size!=item['bytes']:raise ContractError(f'size mismatch: {name}')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        if h.hexdigest()!=item['sha256']:raise ContractError(f'hash mismatch: {name}')
    print(json.dumps({'verified_files':len(seen),'status':'HASHES_MATCH_NOT_RUNTIME_QUALIFICATION'}))
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError) as e:print(str(e),file=sys.stderr);raise SystemExit(2)
