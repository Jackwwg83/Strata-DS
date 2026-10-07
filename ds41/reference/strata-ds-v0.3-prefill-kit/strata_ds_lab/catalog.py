"""Bounded safetensors metadata audit. This is NOT an EXL3 or GGUF model loader."""
from __future__ import annotations
import math
import os
import struct
from pathlib import Path
from .common import ContractError, parse_json, safe_path, require_int
WIDTHS={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,
        'F32':4,'I64':8,'U64':8,'F64':8,'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1,
        'F8_E4M3FN':1}

def inspect_header(path: str | Path, max_header_bytes: int = 64 << 20) -> dict:
    with open(path,'rb') as f:
        size=os.fstat(f.fileno()).st_size
        raw=f.read(8)
        if len(raw)!=8: raise ContractError('truncated safetensors prefix')
        n=struct.unpack('<Q',raw)[0]
        if n < 2 or n > max_header_bytes or n > size-8:
            raise ContractError('invalid or oversized safetensors header')
        header=parse_json(f.read(n).decode('utf-8'))
    if not isinstance(header,dict): raise ContractError('header must be object')
    tensors=[]; spans=[]
    for name,info in header.items():
        if name=='__metadata__':
            if not isinstance(info,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in info.items()):
                raise ContractError('metadata must be string map')
            continue
        if not isinstance(info,dict): raise ContractError('tensor descriptor must be object')
        dtype=info.get('dtype');shape=info.get('shape');off=info.get('data_offsets')
        if dtype not in WIDTHS: raise ContractError(f'unsupported dtype: {dtype}; do not guess widths')
        if not isinstance(shape,list) or len(shape)>16: raise ContractError('invalid shape')
        for d in shape: require_int(d,'dimension')
        if not isinstance(off,list) or len(off)!=2: raise ContractError('invalid offsets')
        a,b=[require_int(x,'offset') for x in off]
        if a>b or b>size-8-n: raise ContractError('tensor outside data section')
        expected=math.prod(shape)*WIDTHS[dtype]
        if b-a!=expected: raise ContractError('shape/dtype/payload mismatch')
        if b>a:spans.append((a,b))
        tensors.append({'name':name,'dtype':dtype,'shape':shape,'bytes':b-a,'file_offset':8+n+a})
    end=0
    for a,b in sorted(spans):
        if a!=end: raise ContractError('overlap or hole in safetensors payload')
        end=b
    if end!=size-8-n: raise ContractError('unindexed trailing bytes')
    return {'file':str(path),'file_bytes':size,'header_bytes':8+n,
            'payload_bytes':sum(t['bytes'] for t in tensors),'tensors':tensors}

def audit_files(root: str | Path, names: list[str]) -> dict:
    if not names:raise ContractError('at least one safetensors file is required')
    root=Path(root);seen=set();resolved=set();files=[]
    for name in names:
        path=safe_path(root,name)
        if path in resolved:raise ContractError('duplicate file alias')
        resolved.add(path)
        data=inspect_header(path)
        for t in data['tensors']:
            if t['name'] in seen: raise ContractError('duplicate tensor name across files')
            seen.add(t['name'])
        data['file']=name;files.append(data)
    return {'status':'HEADERS_ONLY_NOT_CHECKSUM_VERIFIED', 'files':files,
            'payload_bytes':sum(f['payload_bytes'] for f in files),
            'tensor_count':len(seen),
            'missing':'authoritative expert/nonrouted/engram/vision/draft partition and payload hashes'}
