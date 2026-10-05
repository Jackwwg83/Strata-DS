"""Thread-safe cache-control reference. No data storage or CUDA event implementation.
READY publication is authorized by the adapter only after an actual event completes.
Each returned Lease must remain live through the consumer's last GPU/CPU use.
"""
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from .common import ContractError, require_int
@dataclass(frozen=True)
class Ticket:
    key: str
    generation: int
@dataclass
class _Entry:
    nbytes: int
    generation: int
    state: str='LOADING'
    pins: int=0
class Lease:
    def __init__(self,owner,ticket):self._owner=owner;self.ticket=ticket;self._done=False;self._lock=RLock()
    def release(self):
        with self._lock:
            if not self._done:self._owner._release(self.ticket);self._done=True
    def __enter__(self):return self
    def __exit__(self,*args):self.release()
class CacheLedger:
    def __init__(self,capacity_bytes:int):
        self.capacity=require_int(capacity_bytes,'capacity');self.used=0
        self._entries={};self._generation=0;self._lock=RLock()
    def reserve(self,key:str,nbytes:int)->Ticket:
        if not isinstance(key,str) or not key:raise ContractError('empty key')
        require_int(nbytes,'nbytes',1)
        with self._lock:
            if key in self._entries:raise ContractError('duplicate reservation; join existing future instead')
            if self.used+nbytes>self.capacity:raise ContractError('budget exhausted; explicit unpinned eviction required')
            self._generation+=1;t=Ticket(key,self._generation)
            self._entries[key]=_Entry(nbytes,t.generation);self.used+=nbytes;return t
    def _get(self,t):
        e=self._entries.get(t.key)
        if e is None or e.generation!=t.generation:raise ContractError('stale ticket')
        return e
    def publish(self,t:Ticket,*,copy_complete:bool):
        with self._lock:
            e=self._get(t)
            if e.state!='LOADING' or copy_complete is not True:raise ContractError('cannot publish before completed transfer')
            e.state='READY'
    def fail(self,t:Ticket):
        with self._lock:
            e=self._get(t)
            if e.state!='LOADING' or e.pins:raise ContractError('fail is only for unleased loading entries')
            self.used-=e.nbytes;del self._entries[t.key]
    def acquire(self,key:str)->Lease:
        with self._lock:
            e=self._entries.get(key)
            if e is None or e.state!='READY':raise ContractError('entry is not ready')
            e.pins+=1;return Lease(self,Ticket(key,e.generation))
    def _release(self,t):
        with self._lock:
            e=self._get(t)
            if e.pins<=0:raise ContractError('invalid release')
            e.pins-=1
    def evict(self,key:str):
        with self._lock:
            e=self._entries.get(key)
            if e is None: return
            if e.state!='READY' or e.pins:raise ContractError('in-flight/leased entry cannot be evicted')
            self.used-=e.nbytes;del self._entries[key]
