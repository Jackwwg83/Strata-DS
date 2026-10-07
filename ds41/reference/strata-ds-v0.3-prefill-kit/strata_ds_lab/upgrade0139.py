"""Offline v0.1.39 design contracts. No allocations, HTTP, model or cloud execution.

All arithmetic requires caller-supplied measured byte counts. A feasible arithmetic
sum is not a complete memory model or a GPU launch qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, digest_json, read_json, require_int, safe_path

ROOT = Path(__file__).resolve().parents[1]
PHASES = frozenset({'load', 'prefill', 'decode', 'capture', 'transition'})
PIN = '6f32ec070f23ced9f50e704d854d775da52591ab'


def ring_capacity(budget_bytes: int, aligned_bundle_stride_bytes: int,
                  minimum_slots: int = 1, maximum_slots: int | None = None) -> dict[str, Any]:
    """Fit complete bundles in a byte budget; never clamp upward past the budget.

    The stride must already include real format padding and alignment. Kernel
    workspace and any simultaneous host buffer must be budgeted separately.
    """
    require_int(budget_bytes, 'budget_bytes')
    require_int(aligned_bundle_stride_bytes, 'aligned_bundle_stride_bytes', 1)
    require_int(minimum_slots, 'minimum_slots', 1)
    if maximum_slots is not None:
        require_int(maximum_slots, 'maximum_slots', minimum_slots)
    count = budget_bytes // aligned_bundle_stride_bytes
    if maximum_slots is not None:
        count = min(count, maximum_slots)
    if count < minimum_slots:
        raise ContractError('ring cannot fit the algorithm minimum; no upward over-budget clamp')
    allocated = count * aligned_bundle_stride_bytes
    return {'ring_slots': count, 'ring_allocated_bytes': allocated,
            'unspent_budget_bytes': budget_bytes - allocated,
            'evidence': 'CALLER_INPUT_ARITHMETIC_ONLY', 'can_launch': False}


def expert_file_io_advice(*, available_after_residency_bytes: int,
                          future_reserve_bytes: int,
                          file_expert_working_set_bytes: int,
                          direct_io_qualified: bool = False) -> dict[str, Any]:
    """Suggest an A/B candidate using a POST-residency availability snapshot.

    Do not subtract already allocated resident memory again. The caller derives
    the union of possibly read expert ranges, including borrowed-slot refills;
    full Engram table size is not this expert working set. This does not measure
    reclaimability, cgroups, file-cache charge, direct I/O or throughput.
    """
    for name, value in (('available_after_residency_bytes', available_after_residency_bytes),
                        ('future_reserve_bytes', future_reserve_bytes),
                        ('file_expert_working_set_bytes', file_expert_working_set_bytes)):
        require_int(value, name)
    if not isinstance(direct_io_qualified, bool):
        raise ContractError('direct_io_qualified must be a bool backed by external evidence')
    room = max(0, available_after_residency_bytes - future_reserve_bytes)
    if not file_expert_working_set_bytes:
        candidate = 'NO_EXPERT_FILE_READS_IN_THIS_PHASE'
    elif room >= file_expert_working_set_bytes:
        candidate = 'BUFFERED_AB_CANDIDATE'
    elif direct_io_qualified:
        candidate = 'DIRECT_AB_CANDIDATE'
    else:
        candidate = 'BOUNDED_BUFFERED_PRESSURE_TEST_REQUIRED'
    return {'candidate': candidate, 'advisory_room_bytes': room,
            'expert_file_working_set_bytes': file_expert_working_set_bytes,
            'availability_phase': 'AFTER_RESIDENCY', 'can_launch': False,
            'evidence': 'CALLER_INPUT_ADVICE_NOT_PAGE_CACHE_OR_IO_PROOF'}


def phase_peak(*, phase: str, capacity_bytes: int, allocated_slots: int,
               per_slot_bytes: int, co_live_shared_bytes: Mapping[str, int]) -> dict[str, Any]:
    """Sum simultaneously live terms in ONE phase. Unknown values are errors.

    Counts allocated slots even when only one client is active. Shared terms
    must be disjoint and exclude the per-slot term. This function cannot discover
    omitted tensors/allocator overhead, nor prove these inputs are measured.
    """
    if phase not in PHASES:
        raise ContractError('unknown phase')
    require_int(capacity_bytes, 'capacity_bytes', 1)
    require_int(allocated_slots, 'allocated_slots', 1)
    require_int(per_slot_bytes, 'per_slot_bytes')
    if not isinstance(co_live_shared_bytes, Mapping) or not co_live_shared_bytes:
        raise ContractError('co_live_shared_bytes must be a nonempty mapping of disjoint byte terms')
    total = allocated_slots * per_slot_bytes
    for name, value in co_live_shared_bytes.items():
        if not isinstance(name, str) or not name:
            raise ContractError('phase terms need nonempty names')
        total += require_int(value, name)
    return {'phase': phase, 'supplied_peak_bytes': total,
            'allocated_slots': allocated_slots,
            'per_slot_total_bytes': allocated_slots * per_slot_bytes,
            'capacity_bytes': capacity_bytes, 'arithmetic_fits': total <= capacity_bytes,
            'excess_bytes': max(0, total - capacity_bytes), 'can_launch': False,
            'evidence': 'CALLER_INPUT_ARITHMETIC_NOT_COMPLETE_ALLOCATION_PROOF'}


def require_trial_slots(requested_slots: int, effective_slots: int) -> None:
    """Strict experiment contract: silent engine down-clamping invalidates a trial."""
    require_int(requested_slots, 'requested_slots', 1)
    require_int(effective_slots, 'effective_slots', 1)
    if requested_slots != effective_slots:
        raise ContractError('CAPACITY_MISMATCH: record requested/effective separately, not a successful target trial')


def merged_task_order(base_tasks: list[dict], additions: list[dict],
                      augment: Mapping[str, list[str]]) -> list[str]:
    """Validate both ID spaces and return a stable topological plan (not execution)."""
    nodes: dict[str, set[str]] = {}
    for task in base_tasks + additions:
        tid = task.get('id')
        if not isinstance(tid, str) or not tid or tid in nodes:
            raise ContractError('missing or duplicate task ID')
        deps = task.get('depends_on')
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ContractError('depends_on must be a list of task IDs')
        if len(deps) != len(set(deps)):
            raise ContractError('duplicate dependency')
        nodes[tid] = set(deps)
    for tid, deps in augment.items():
        if tid not in nodes or not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ContractError('invalid dependency augmentation')
        nodes[tid].update(deps)
    for tid, deps in nodes.items():
        if tid in deps or not deps.issubset(nodes):
            raise ContractError('unknown dependency or self cycle')
    result: list[str] = []
    todo = {key: set(value) for key, value in nodes.items()}
    while todo:
        ready = sorted(key for key, deps in todo.items() if not deps)
        if not ready:
            raise ContractError('task dependency cycle')
        for key in ready:
            del todo[key]
            result.append(key)
        for deps in todo.values():
            deps.difference_update(ready)
    return result


def validate_profiles(data: dict) -> list[dict]:
    if data.get('default_requested_slots') != 1 or data.get('ds_session_bytes') is not None:
        raise ContractError('default must remain one slot; no invented DeepSeek per-slot size')
    if data.get('phase_memory_measurement_required') is not True:
        raise ContractError('phase memory measurements must gate real execution')
    profiles = data.get('profiles')
    if not isinstance(profiles, list) or len(profiles) != 8:
        raise ContractError('expected eight reviewed runtime profiles')
    ids: set[str] = set()
    for p in profiles:
        pid = p.get('id')
        if not isinstance(pid, str) or not re.fullmatch(r'[a-z0-9-]+', pid) or pid in ids:
            raise ContractError('invalid or duplicate profile ID')
        ids.add(pid)
        require_int(p.get('requested_slots'), 'requested_slots', 1)
        require_int(p.get('offered_concurrency'), 'offered_concurrency', 1)
        if (p.get('can_launch') is not False or p.get('speculation') is not False
                or p.get('vision') is not False or p.get('prefill_borrow') is not False
                or p.get('readiness') != 'PLANNED_NOT_IMPLEMENTED'):
            raise ContractError('profiles must stay unqualified plans without optional execution changes')
    initial = data.get('initial_profiles')
    if initial != ['serial-fixed', 'serial-byte'] or not set(initial).issubset(ids):
        raise ContractError('initial scope must remain the two single-slot profiles')
    return profiles


def validate_design(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    preserve = read_json(root / 'specs/upstream/v02-preserved.json')
    for item in preserve['files']:
        path = safe_path(root, item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ContractError(f'baseline drift: {item["path"]}; review instead of silently overwriting')
    lock = read_json(root / 'specs/sources.lock.json')
    for key, value in preserve['preserved_source_lock_keys'].items():
        if lock.get(key) != value:
            raise ContractError(f'model/source baseline changed: {key}')
    if lock.get('strata_reference', {}).get('commit') != PIN:
        raise ContractError('Strata reference SHA mismatch')
    cloud = read_json(root / 'specs/cloud-policy.json')
    if cloud['allow_paid_create'] is not False or cloud['allow_model_download'] is not False:
        raise ContractError('design kit must not authorize cloud writes/downloads')
    profiles = validate_profiles(read_json(root / 'specs/upstream-v0139.profiles.json'))
    base = read_json(root / 'specs/backlog.json')
    delta = read_json(root / 'specs/upstream-v0139.backlog.json')
    order = merged_task_order(base['tasks'], delta['tasks'], delta['augment_existing_prerequisites'])
    if len(base['tasks']) != 18 or len(delta['tasks']) != 12:
        raise ContractError('reviewed task count drift')
    if any(x['status'] != 'planned' for x in base['tasks'] + delta['tasks']):
        raise ContractError('offline validation is not task implementation')
    return {'status': 'DESIGN_CONTRACTS_VALID_NOT_IMPLEMENTATION', 'can_launch': False,
            'preserved_recipe_count': 11, 'delta_tasks': len(delta['tasks']),
            'merged_task_count': len(order), 'runtime_profiles': len(profiles),
            'topological_task_order': order, 'strata_reference_commit': PIN,
            'model_pins_changed': False, 'paid_actions_enabled': False}


def planned_matrix(root: Path, recipe_id: str, initial_only: bool = False,
                   repeats: int = 3) -> list[dict[str, Any]]:
    validate_design(root)
    require_int(repeats, 'repeats', 1)
    if repeats > 20:
        raise ContractError('offline matrix cap is 20 repeats; review broader trials explicitly')
    if not re.fullmatch(r'[a-z0-9.-]+', recipe_id):
        raise ContractError('invalid recipe ID')
    recipe = read_json(safe_path(root, f'recipes/{recipe_id}.json'))
    cfg = read_json(root / 'specs/upstream-v0139.profiles.json')
    selected = [p for p in cfg['profiles'] if not initial_only or p['id'] in cfg['initial_profiles']]
    rows = []
    for profile in selected:
        for ctx in profile['context_targets']:
            for repeat in range(1, repeats + 1):
                row = {'evidence': 'DESIGN_PLAN_NOT_EXECUTED', 'recipe_id': recipe_id,
                       'recipe_sha256': digest_json(recipe), 'profile_id': profile['id'],
                       'profile_sha256': digest_json(profile), 'requested_slots': profile['requested_slots'],
                       'effective_slots': None, 'offered_concurrency': profile['offered_concurrency'],
                       'context_target': ctx, 'exact_prompt_tokens': None, 'trial_repeat': repeat,
                       'load_pattern': profile['load_pattern'], 'runtime_profile': profile,
                       'can_launch': False, 'hardware_preflight': None,
                       'model_correctness_evidence': None, 'paid_approval': None}
                row['plan_id'] = digest_json(row)[:20]
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('validate')
    m = sub.add_parser('matrix')
    m.add_argument('--recipe', required=True)
    m.add_argument('--initial-only', action='store_true')
    m.add_argument('--repeats', type=int, default=3)
    a = p.parse_args(argv)
    try:
        if a.command == 'validate':
            print(json.dumps(validate_design(a.root), ensure_ascii=False, indent=2))
        else:
            for row in planned_matrix(a.root, a.recipe, a.initial_only, a.repeats):
                print(json.dumps(row, ensure_ascii=False, allow_nan=False))
        return 0
    except (ContractError, OSError, KeyError, TypeError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
