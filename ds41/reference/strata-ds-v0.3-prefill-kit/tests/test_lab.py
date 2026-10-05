from __future__ import annotations
import copy
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from strata_ds_lab.common import ContractError,GiB,read_json,parse_json,safe_path,digest_json
from strata_ds_lab.planner import plan,traffic_ceiling
from strata_ds_lab.catalog import inspect_header,audit_files
from strata_ds_lab.rowstore import FileRows
from strata_ds_lab.leases import CacheLedger
from strata_ds_lab.contracts import require_backend,BACKENDS
from strata_ds_lab.matrix import make_cases
from strata_ds_lab.metrics import summarize,percentile
from strata_ds_lab.preflight import parse_meminfo,inspect_cgroup_chain
from strata_ds_lab.vastplan import search_argv,cost_plan
from strata_ds_lab.cli import main
ROOT=Path(__file__).resolve().parents[1]
def recipe(name='exl3-3-r128-v24'):return read_json(ROOT/'recipes'/f'{name}.json')
def model(name='exl3-3'):return read_json(ROOT/'specs/models.json')['models'][name]
def fixture(path,header=None,data=b'abcdefgh'):
    if header is None:header={'x':{'dtype':'U8','shape':[8],'data_offsets':[0,8]}}
    h=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(h))+h+data)
def records():return [json.loads(l) for l in (ROOT/'examples/timings.synthetic.jsonl').read_text().splitlines()]
class CommonTests(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaises(ContractError):parse_json('{"x":1,"x":2}')
    def test_nonfinite_json(self):
        with self.assertRaises(ContractError):parse_json('{"x":NaN}')
    def test_bounded_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('{} ')
            with self.assertRaises(ContractError):read_json(p,2)
    def test_path_traversal(self):
        with self.assertRaises(ContractError):safe_path(ROOT,'../outside')
    def test_absolute_path(self):
        with self.assertRaises(ContractError):safe_path(ROOT,'/etc/passwd')
    def test_canonical_digest(self):self.assertEqual(digest_json({'b':1,'a':2}),digest_json({'a':2,'b':1}))
class PlannerTests(unittest.TestCase):
    def test_all_11_design_only(self):
        rs=list((ROOT/'recipes').glob('*.json'));self.assertEqual(len(rs),11)
        for p in rs:
            with self.subTest(p=p.name):
                r=read_json(p);x=plan(r,model(r['model']));self.assertFalse(x['can_launch'])
    def test_exl3_128_spill(self):
        p=plan(recipe(),model());self.assertEqual(p['host_weight_ceiling_bytes'],96*GiB)
        self.assertAlmostEqual(p['cold_weight_lower_bound_gib'],86.555510938,places=6)
    def test_exl3_192_spill(self):self.assertAlmostEqual(plan(recipe('exl3-3-r192-v24'),model())['cold_weight_lower_bound_gib'],22.555510938,places=6)
    def test_exl3_256_possible_not_certified(self):
        p=plan(recipe('exl3-3-r256-v24'),model());self.assertTrue(p['possible_all_active_weight_residency']);self.assertFalse(p['can_launch'])
    def test_separate_gpu_pool(self):
        r=recipe();r['vram_gib']=6
        with self.assertRaises(ContractError):plan(r,model())
    def test_unknown_topology(self):
        r=recipe();r['topology']='unified'
        with self.assertRaises(ContractError):plan(r,model())
    def test_missing_reserve(self):
        r=recipe();del r['host_reserve_gib']['engram_rows']
        with self.assertRaises(ContractError):plan(r,model())
    def test_bool_not_memory(self):
        r=recipe();r['ram_gib']=True
        with self.assertRaises(ContractError):plan(r,model())
    def test_h2d_even_when_ram_hit_one(self):
        r=traffic_ceiling(3_000_000_000,0,1,3e9,12e9)
        self.assertEqual(r['ssd_logical_bytes_per_token'],0);self.assertEqual(r['h2d_packed_bytes_per_token'],3e9)
        self.assertEqual(r['tokens_per_second_upper_bound'],4)
    def test_bad_hit(self):
        with self.assertRaises(ContractError):traffic_ceiling(100,1.1,1,1e9,1e9)
    def test_no_cpu_assumption(self):
        with self.assertRaises(ContractError):traffic_ceiling(100,0,1,1e9,1e9,execution='cpu')
class CatalogTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.p=self.root/'a.safetensors'
    def tearDown(self):self.t.cleanup()
    def test_valid(self):fixture(self.p);self.assertEqual(inspect_header(self.p)['payload_bytes'],8)
    def test_short_prefix(self):
        self.p.write_bytes(b'bad')
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_header_bomb(self):
        self.p.write_bytes(struct.pack('<Q',1<<50))
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_wrong_shape(self):
        fixture(self.p,{'x':{'dtype':'U8','shape':[9],'data_offsets':[0,8]}})
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_unknown_dtype(self):
        fixture(self.p,{'x':{'dtype':'MADE_UP','shape':[8],'data_offsets':[0,8]}})
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_overlap(self):
        fixture(self.p,{'x':{'dtype':'U8','shape':[8],'data_offsets':[0,8]},'y':{'dtype':'U8','shape':[1],'data_offsets':[0,1]}})
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_trailing(self):
        fixture(self.p,data=b'abcdefghx')
        with self.assertRaises(ContractError):inspect_header(self.p)
    def test_empty_tensor(self):
        fixture(self.p,{'x':{'dtype':'F32','shape':[0],'data_offsets':[0,0]}},b'')
        self.assertEqual(inspect_header(self.p)['payload_bytes'],0)
    def test_cross_file_duplicate(self):
        fixture(self.p);fixture(self.root/'b.safetensors')
        with self.assertRaises(ContractError):audit_files(self.root,['a.safetensors','b.safetensors'])
    def test_symlink_escape(self):
        (self.root/'out.safetensors').symlink_to('/etc/passwd')
        with self.assertRaises(ContractError):audit_files(self.root,['out.safetensors'])
class RowsTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.p=Path(self.t.name)/'rows';self.p.write_bytes(b'HEAD'+b'ab--cd--ef--')
    def tearDown(self):self.t.cleanup()
    def test_row_order_and_stride(self):
        with FileRows(self.p,4,3,2,4) as r:self.assertEqual(r.read_rows([2,0,1]),[b'ef',b'ab',b'cd'])
    def test_lazy_open(self):
        with patch('os.pread',side_effect=AssertionError('no eager data read')):
            r=FileRows(self.p,4,3,2,4);r.close()
    def test_dedup(self):
        with FileRows(self.p,4,3,2,4) as r:
            r.read_rows([0,0,0]);self.assertEqual(r.stats['pread_calls'],1);self.assertEqual(r.stats['batch_dedup_hits'],2)
    def test_bounded_cache(self):
        with FileRows(self.p,4,3,2,4,cache_bytes=4) as r:
            r.read_rows([0,1,2]);self.assertEqual(r.cache_payload_bytes,4)
            r.read_rows([2]);self.assertEqual(r.stats['cache_hits'],1)
    def test_invalid_index(self):
        with FileRows(self.p,4,3,2,4) as r:
            for ids in [[-1],[3],[True]]:
                with self.subTest(ids=ids),self.assertRaises(ContractError):r.read_rows(ids)
    def test_file_truncated_after_open(self):
        with FileRows(self.p,4,3,2,4) as r:
            self.p.write_bytes(b'x')
            with self.assertRaises(ContractError):r.read_rows([0])
    def test_short_read_failure(self):
        with FileRows(self.p,4,3,2,4) as r,patch('os.pread',return_value=b'a'):
            with self.assertRaises(ContractError):r.read_rows([0])
    def test_closed(self):
        r=FileRows(self.p,4,3,2,4);r.close();r.close()
        with self.assertRaises(ContractError):r.read_rows([0])
    def test_output_budget(self):
        with self.assertRaises(ContractError):FileRows(self.p,0,1,1<<20,cache_bytes=0,max_batch_rows=17)
    def test_threadsafe(self):
        with FileRows(self.p,4,3,2,4,cache_bytes=6) as r,ThreadPoolExecutor(4) as pool:
            self.assertTrue(all(x==[b'cd'] for x in pool.map(lambda _:r.read_rows([1]),range(20))))
class LeaseTests(unittest.TestCase):
    def test_publish_requires_completion(self):
        c=CacheLedger(10);t=c.reserve('x',5)
        with self.assertRaises(ContractError):c.publish(t,copy_complete=False)
        with self.assertRaises(ContractError):c.acquire('x')
    def test_lease_blocks_eviction(self):
        c=CacheLedger(10);t=c.reserve('x',5);c.publish(t,copy_complete=True)
        with c.acquire('x'):
            with self.assertRaises(ContractError):c.evict('x')
        c.evict('x');self.assertEqual(c.used,0)
    def test_generation_fencing(self):
        c=CacheLedger(10);old=c.reserve('x',5);c.fail(old);new=c.reserve('x',5)
        self.assertNotEqual(old.generation,new.generation)
        with self.assertRaises(ContractError):c.publish(old,copy_complete=True)
    def test_reservation_budget(self):
        c=CacheLedger(10);c.reserve('x',8)
        with self.assertRaises(ContractError):c.reserve('y',3)
    def test_no_implicit_evict_loading(self):
        c=CacheLedger(10);c.reserve('x',8)
        with self.assertRaises(ContractError):c.evict('x')
    def test_release_idempotent(self):
        c=CacheLedger(10);t=c.reserve('x',5);c.publish(t,copy_complete=True);l=c.acquire('x');l.release();l.release();c.evict('x')
    def test_double_publish_rejected(self):
        c=CacheLedger(10);t=c.reserve('x',5);c.publish(t,copy_complete=True)
        with self.assertRaises(ContractError):c.publish(t,copy_complete=True)
    def test_parallel_reservations_bounded(self):
        c=CacheLedger(10)
        def f(i):
            try:c.reserve(str(i),1);return 1
            except ContractError:return 0
        with ThreadPoolExecutor(8) as pool:self.assertEqual(sum(pool.map(f,range(100))),10)
class MatrixTests(unittest.TestCase):
    def test_smoke_not_measurement(self):
        c=make_cases([recipe()]);self.assertEqual(len(c),1);self.assertEqual(c[0]['status'],'PLANNED_NOT_EXECUTED')
    def test_screen_count(self):self.assertEqual(len(make_cases([recipe()],'screen')),18)
    def test_full_count_and_ids(self):
        c=make_cases([recipe()],'full');self.assertEqual(len(c),72);self.assertEqual(len(set(x['case_id'] for x in c)),72)
    def test_token_budget(self):
        for x in make_cases([recipe()],'full'):self.assertLessEqual(x['prompt_target_tokens']+x['output_token_limit'],x['context_allocated_tokens'])
    def test_duplicate_recipe_rejected(self):
        with self.assertRaises(ContractError):make_cases([recipe(),recipe()])
class MetricsTests(unittest.TestCase):
    def test_aggregate_wall_not_sum_rates(self):self.assertEqual(summarize(records())['aggregate_goodput_tok_s'],1)
    def test_provenance_retained(self):self.assertEqual(summarize(records())['provenance'],'synthetic_test')
    def test_failed_time_in_denominator(self):
        r=records();r[1]['status']='timeout';self.assertAlmostEqual(summarize(r)['aggregate_goodput_tok_s'],.6)
    def test_mixed_condition_rejected(self):
        r=records();r[1]['condition_sha256']='other'
        with self.assertRaises(ContractError):summarize(r)
    def test_chunk_not_token(self):
        r=records();r[0]['timing_source']='sse_chunk'
        with self.assertRaises(ContractError):summarize(r)
    def test_duplicate_id(self):
        r=records();r[1]['request_id']='a'
        with self.assertRaises(ContractError):summarize(r)
    def test_count_mismatch(self):
        r=records();r[0]['output_tokens']=99
        with self.assertRaises(ContractError):summarize(r)
    def test_order_rejected(self):
        r=records();r[0]['token_timestamps_ns']=[3,2,1]
        with self.assertRaises(ContractError):summarize(r)
    def test_percentile(self):self.assertEqual(percentile([0,10],.5),5)
class CloudAndPreflightTests(unittest.TestCase):
    def test_search_only_argv(self):
        a=search_argv(recipe());self.assertEqual(a[:3],['vastai','search','offers']);self.assertNotIn('create',a)
    def test_cost_disabled(self):
        q=read_json(ROOT/'examples/normalized-quote.synthetic.json');p=read_json(ROOT/'specs/cloud-policy.json');v=cost_plan(q,p)
        self.assertFalse(v['can_execute']);self.assertTrue(v['blockers']);self.assertAlmostEqual(v['estimated_total_usd'],6.27)
    def test_cost_not_double_count_disk(self):
        q=read_json(ROOT/'examples/normalized-quote.synthetic.json');q.update(download_gb=0,upload_gb=0)
        self.assertEqual(cost_plan(q,{})['estimated_total_usd'],2)
    def test_even_approval_does_not_execute(self):
        q=read_json(ROOT/'examples/normalized-quote.synthetic.json');p=dict(max_hourly_usd=2,max_total_usd=10,max_duration_hours=4,allow_paid_create=True,max_active_instances=1)
        v=cost_plan(q,p);self.assertFalse(v['blockers']);self.assertFalse(v['can_execute'])
    def test_meminfo_units(self):self.assertEqual(parse_meminfo('MemTotal: 1024 kB')['MemTotal'],1<<20)
    def test_ancestor_tighter_than_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);child=root/'a';child.mkdir()
            (root/'memory.max').write_text('100');(root/'memory.current').write_text('90')
            (child/'memory.max').write_text('200');(child/'memory.current').write_text('10')
            v=inspect_cgroup_chain(child,root);self.assertEqual(v['remaining_bytes'],10)
    def test_unknown_cgroup_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            v=inspect_cgroup_chain(Path(d),Path(d));self.assertFalse(v['complete']);self.assertIsNone(v['remaining_bytes'])
class CLITests(unittest.TestCase):
    def test_all_real_backends_blocked(self):
        for name in BACKENDS:
            with self.subTest(name=name),self.assertRaisesRegex(ContractError,'BACKEND_NOT_IMPLEMENTED'):require_backend(name)
    def test_cli_plan(self):
        with patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(main(['plan','exl3-3-r256-v24']),0);self.assertFalse(json.loads(out.getvalue())['can_launch'])
    def test_cli_bad_path(self):
        with patch('sys.stderr',new_callable=io.StringIO):self.assertEqual(main(['plan','../secret']),2)
    def test_cli_vast_never_executes(self):
        with patch('subprocess.run',side_effect=AssertionError('must not execute')),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(main(['vast-search','exl3-3-r128-v24']),0)

class ExtraBoundaryTests(unittest.TestCase):
    def test_bool_schema_rejected(self):
        r=recipe();r['schema_version']=True
        with self.assertRaises(ContractError):plan(r,model())
    def test_native_lower_bound_not_fit(self):
        x=plan(recipe('native-r384-v24'),model('native'))
        self.assertIsNone(x['possible_all_active_weight_residency'])
    def test_empty_audit_rejected(self):
        with self.assertRaises(ContractError):audit_files(ROOT,[])
    def test_smoke_not_circular_gate(self):
        c=make_cases([recipe()],'smoke')[0]
        self.assertIn('operator_parity',c['required_gates'])
        self.assertNotIn('same_quant_offload_parity',c['required_gates'])
    def test_parallel_release_idempotent(self):
        c=CacheLedger(10);t=c.reserve('x',5);c.publish(t,copy_complete=True);l=c.acquire('x')
        with ThreadPoolExecutor(8) as pool:list(pool.map(lambda _:l.release(),range(20)))
        c.evict('x');self.assertEqual(c.used,0)

if __name__=='__main__':unittest.main()
