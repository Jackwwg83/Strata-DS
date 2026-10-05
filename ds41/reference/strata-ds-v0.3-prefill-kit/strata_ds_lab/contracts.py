"""Interface boundaries for a future real backend. All GPU adapters remain missing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
from .common import ContractError
@dataclass(frozen=True)
class ModelIdentity:
    base_revision: str
    quant_manifest_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    engram_semantics_sha256: str
@dataclass(frozen=True)
class PackedExpert:
    layer: int
    expert: int
    quant_format: str
    payload_bytes: int
    component_names: tuple[str,...]
class Completion(Protocol):
    def wait(self,timeout_s:float)->None:...
    def completed(self)->bool:...
class ExpertBackend(Protocol):
    def validate(self,identity:ModelIdentity,manifest:dict)->None:...
    def stage(self,expert:PackedExpert,device_slot:int)->Completion:...
    def forward(self,activations,experts:Sequence[PackedExpert],routing_weights):...
class EngineAdapter(Protocol):
    def describe_capabilities(self)->dict:...
    def load(self,plan:dict,manifest:dict)->None:...
    def prefill(self,session_id:str,token_ids:Sequence[int],expected_position:int):...
    def decode(self,session_ids:Sequence[str]):...
    def snapshot(self,session_id:str)->dict:...
    def close(self)->None:...
BACKENDS={
  'exl3_gpu_stream':{'implemented':False,'reason':'single discrete GPU EXL3 V4.1 bridge required; TP2/GB10 image is not interchangeable'},
  'gguf_q2_bridge':{'implemented':False,'reason':'pinned DwarfStar subprocess/metrics bridge and discrete hybrid qualification required'},
  'native_oracle':{'implemented':False,'reason':'independent fixed-native oracle not integrated'},
  'sage_exl3_adapter':{'implemented':False,'reason':'mixed K/layout support and independent quality gate required'}
}
def require_backend(name:str):
    if name not in BACKENDS:raise ContractError('unknown backend')
    if not BACKENDS[name]['implemented']:
        raise ContractError('BACKEND_NOT_IMPLEMENTED: '+BACKENDS[name]['reason'])
    return BACKENDS[name]
