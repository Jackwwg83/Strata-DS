"""Portable synchronous reference for bounded SSD rows. No direct-I/O or CUDA claim.
The cache bounds payload bytes, NOT Python overhead, return buffers or OS page cache.
Backend pins/validates a local immutable file generation before using this reader.
"""
from __future__ import annotations
import os
import stat
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from .common import ContractError, require_int
class FileRows:
    def __init__(self,path: str | Path,offset:int,rows:int,row_bytes:int,
                 stride:int|None=None,cache_bytes:int=0,max_batch_rows:int=1024):
        self.offset=require_int(offset,'offset')
        self.rows=require_int(rows,'rows',1)
        self.row_bytes=require_int(row_bytes,'row_bytes',1)
        self.stride=require_int(row_bytes if stride is None else stride,'stride',1)
        if self.stride<self.row_bytes: raise ContractError('stride smaller than row')
        self.cache_bytes=require_int(cache_bytes,'cache bytes')
        self.max_batch_rows=require_int(max_batch_rows,'max batch rows',1)
        if self.row_bytes>1<<20: raise ContractError('row too large for this reference reader')
        if self.max_batch_rows*self.row_bytes>16<<20: raise ContractError('batch return buffer exceeds 16 MiB')
        self._fd=os.open(path,os.O_RDONLY)
        self._lock=RLock();self._cache=OrderedDict();self.cache_payload_bytes=0
        self.stats={'logical_rows':0,'cache_hits':0,'batch_dedup_hits':0,'pread_calls':0,'pread_bytes':0}
        st=os.fstat(self._fd)
        if not stat.S_ISREG(st.st_mode) or self.offset+(rows-1)*self.stride+self.row_bytes>st.st_size:
            os.close(self._fd);self._fd=-1;raise ContractError('rows outside regular file')
        self._identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    def _check(self):
        if self._fd<0: raise ContractError('reader closed')
        st=os.fstat(self._fd)
        if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)!=self._identity:
            raise ContractError('file changed while reader was open')
    def read_rows(self,ids:list[int])->list[bytes]:
        with self._lock:
            self._check()
            if len(ids)>self.max_batch_rows: raise ContractError('too many rows; chunk the request')
            for r in ids:
                require_int(r,'row')
                if r>=self.rows:raise ContractError('row out of range; never silently zero-fill')
            local={}
            for r in ids:
                self.stats['logical_rows']+=1
                if r in local:
                    self.stats['batch_dedup_hits']+=1;continue
                if r in self._cache:
                    b=self._cache.pop(r);self._cache[r]=b;self.stats['cache_hits']+=1
                else:
                    b=os.pread(self._fd,self.row_bytes,self.offset+r*self.stride)
                    self.stats['pread_calls']+=1;self.stats['pread_bytes']+=len(b)
                    if len(b)!=self.row_bytes: raise ContractError('short read')
                    if self.cache_bytes>=self.row_bytes:
                        while self.cache_payload_bytes+self.row_bytes>self.cache_bytes:
                            self._cache.popitem(last=False);self.cache_payload_bytes-=self.row_bytes
                        self._cache[r]=b;self.cache_payload_bytes+=self.row_bytes
                local[r]=b
            return [local[r] for r in ids]
    def close(self):
        with self._lock:
            if self._fd>=0:os.close(self._fd);self._fd=-1
            self._cache.clear();self.cache_payload_bytes=0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
