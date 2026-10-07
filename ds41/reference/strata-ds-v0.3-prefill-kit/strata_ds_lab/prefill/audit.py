"""Local, header-and-marker-only EXL3 audit. Never allocates expert payloads.
Payload hashes and GPU numerical qualification remain separate gates.
"""
from __future__ import annotations
import hashlib
import os
import re
import struct
from pathlib import Path
from ..common import ContractError, safe_path, digest_json
from ..catalog import audit_files
from .layout import Geometry, components, bundle_layout, MUL1
PATTERN=re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(trellis|suh|svh|mul1)$')

def _signature(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def audit_exl3(root: str|Path, names: list[str], *, complete: bool=False,
               geometry: Geometry=Geometry(), read_markers: bool=True) -> dict:
    root=Path(root).resolve()
    snapshots={name:_signature(safe_path(root,name)) for name in names}
    catalog=audit_files(root,names)
    expected={(c.projection,c.suffix):c for c in components(geometry)}
    groups={}; other=0
    for f in catalog['files']:
        path=safe_path(root,f['file'])
        with path.open('rb') as stream:
            fdstat=os.fstat(stream.fileno())
            if (fdstat.st_dev,fdstat.st_ino,fdstat.st_size,fdstat.st_mtime_ns,fdstat.st_ctime_ns)!=snapshots[f['file']]:
                raise ContractError('file changed before marker read')
            for t in f['tensors']:
                m=PATTERN.fullmatch(t['name'])
                if m is None:
                    if '.ffn.experts.' in t['name'] and t['name'].startswith('layers.'):
                        raise ContractError(f'unknown backbone expert component {t["name"]}')
                    other+=t['bytes'];continue
                l,e=int(m[1]),int(m[2]);key=(m[3],m[4])
                if l>=geometry.layers or e>=geometry.experts:
                    raise ContractError('expert/layer ID outside geometry')
                c=expected[key]
                shape=tuple(t['shape'])
                if c.suffix=='mul1' and shape==():shape=(1,)
                if shape!=c.shape or t['dtype']!=c.dtype or t['bytes']!=c.payload_bytes:
                    raise ContractError(f'packed schema mismatch: {t["name"]}')
                entry={**t,'file':f['file'],'slot_offset':c.slot_offset}
                if c.suffix=='mul1' and read_markers:
                    stream.seek(t['file_offset']);raw=stream.read(4)
                    if len(raw)!=4 or struct.unpack('<I',raw)[0]!=MUL1:
                        raise ContractError(f'MUL1 marker mismatch: {t["name"]}')
                    entry['marker_u32']=MUL1
                group=groups.setdefault((l,e),{})
                if key in group:raise ContractError('duplicate expert component')
                group[key]=entry
    if not groups:raise ContractError('no complete-backbone EXL3 experts found')
    bundles=[]
    for (l,e),parts in sorted(groups.items()):
        if parts.keys()!=expected.keys():raise ContractError(f'incomplete expert bundle {l}/{e}')
        bundles.append({'layer':l,'expert':e,'payload_bytes':sum(p['bytes'] for p in parts.values()),
                        'components':[parts[k] for k in expected]})
    if complete and len(bundles)!=geometry.layers*geometry.experts:
        raise ContractError('full backbone requested but experts are missing')
    for name,signature in snapshots.items():
        if _signature(safe_path(root,name))!=signature:raise ContractError('file changed during audit')
    return {'status':'LOCAL_HEADERS_AND_MARKERS_VERIFIED' if read_markers else 'LOCAL_HEADERS_VERIFIED_MARKERS_UNCHECKED',
            'full_expert_coverage':len(bundles)==geometry.layers*geometry.experts,
            'payload_sha256_verified':False,'gpu_qualified':False,
            'expert_count':len(bundles),'routed_payload_bytes':sum(b['payload_bytes'] for b in bundles),
            'unclassified_nonexpert_payload_bytes':other,'unclassified_is_not_gpu_mandatory':True,
            'layout':bundle_layout(geometry),'bundle_descriptor_digest':digest_json(bundles),'bundles':bundles}
