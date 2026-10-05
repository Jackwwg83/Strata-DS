from __future__ import annotations
import contextlib
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
from strata_ds_lab.common import ContractError,read_json
from strata_ds_lab.prefill.layout import Geometry,components,bundle_layout,ring_layout,align,load_facts,validate_target_config,ROOT,MUL1,MiB,GiB
from strata_ds_lab.prefill.planner import Inputs,workspace_budget,state_budget,make_plan,seed_for_recipe,candidate_plans
from strata_ds_lab.prefill.routes import group_routes,make_waves,execute_reference
from strata_ds_lab.prefill.audit import audit_exl3
from strata_ds_lab.prefill.tune import choose_measured

class LayoutTests(unittest.TestCase):
    def test_source_geometry(self):
        f=load_facts();self.assertEqual((f['layers'],f['experts'],f['top_k']), (40,384,6))
    def test_real_tp1_shapes(self):
        c=components();self.assertEqual(c[0].shape,(320,144,48));self.assertEqual(c[8].shape,(144,320,48))
    def test_payload(self):self.assertEqual(bundle_layout()['source_payload_bytes'],13315596)
    def test_component_bytes(self):
        c=components();self.assertEqual([x.payload_bytes for x in c[:4]],[4423680,10240,4608,4])
    def test_stride(self):self.assertEqual(bundle_layout()['slot_stride_bytes'],13316352)
    def test_component_alignment(self):
        for c in components():self.assertEqual(c.slot_offset%256,0)
    def test_no_component_overlap(self):
        cs=components()
        for a,b in zip(cs,cs[1:]):self.assertLessEqual(a.slot_offset+a.payload_bytes,b.slot_offset)
    def test_exact_bank(self):self.assertEqual(bundle_layout()['full_routed_bank_payload_bytes'],204527554560)
    def test_reconstruct(self):self.assertEqual(bundle_layout()['one_fp16_reconstructed_matrix_bytes'],23592960)
    def test_ring_counts(self):
        for size,slots in [(256,20),(512,40),(1024,80)]:
            self.assertEqual(ring_layout(size*MiB,4)['slots'],slots)
    def test_no_min_clamp(self):self.assertFalse(ring_layout(1,1)['feasible'])
    def test_double_wave_bound(self):
        s=bundle_layout()['slot_stride_bytes']
        self.assertFalse(ring_layout(7*s,4)['feasible']);self.assertTrue(ring_layout(8*s,4)['feasible'])
    def test_qwen_dimension_rejected(self):
        with self.assertRaises(ContractError):Geometry(hidden=2560,intermediate=640,top_k=513)
    def test_hadamard_misalignment(self):
        with self.assertRaises(ContractError):Geometry(intermediate=2305)
    def test_unknown_quant_rejected(self):
        for b in (1,2,5,True):
            with self.assertRaises(ContractError):Geometry(bits=b)
    def test_four_bit_not_silently_three(self):
        self.assertGreater(bundle_layout(Geometry(bits=4))['source_payload_bytes'],bundle_layout()['source_payload_bytes'])
    def test_alignment_invalid(self):
        for n,q in [(-1,256),(2,0),(1,3),(True,256)]:
            with self.assertRaises(ContractError):align(n,q)
    def test_config_rejects_missing(self):
        with self.assertRaises(ContractError):validate_target_config({'hidden_size':5120})

class BudgetTests(unittest.TestCase):
    def test_bf16_four_banks(self):
        s=state_budget(131072);self.assertEqual(len(s['source_banks']),4)
        self.assertEqual(s['global_bytes_per_session'],131072*3200)
    def test_packed_four_banks(self):
        s=state_budget(131072,1,'packed_890_experimental');self.assertEqual(s['global_bytes_per_session'],131072*890)
    def test_partial_pair(self):
        s=state_budget(129);self.assertEqual(s['source_banks'][0]['logical_rows'],64)
        self.assertEqual(s['source_banks'][-1]['logical_rows'],129)
    def test_page_rounding(self):
        self.assertEqual(state_budget(1)['source_banks'][-1]['allocated_rows'],256)
    def test_slot_cost(self):
        one=state_budget(8192,1)['total_bytes'];self.assertEqual(state_budget(8192,5)['total_bytes'],5*one)
    def test_no_query_sources_as_key_banks(self):
        s=state_budget(1048576);self.assertEqual([x['source_layer'] for x in s['source_banks']],[2,8,14,20])
    def test_unknown_kv(self):
        with self.assertRaises(ContractError):state_budget(8,kv_format='qwen_qsa')
    def test_source_tp2_scratch_reproduced(self):
        rows=1056*6
        self.assertEqual(rows*(4*5120+2*1152+16)+(384+math.ceil(rows/64))*12,144466596)
    def test_tp1_is_not_tp2(self):
        rows=2048*6
        self.assertEqual(rows*(4*5120+2*2304+16)+(384+math.ceil(rows/64))*12,308484864)
    def test_seed_16(self):
        p=seed_for_recipe({'model':'exl3-3','vram_gib':16,'ram_gib':128})
        self.assertEqual((p.chunk,p.ring_mib,p.wave_experts,p.expert_row_tile),(1024,256,4,64))
    def test_seed_24(self):
        p=seed_for_recipe({'model':'exl3-3','vram_gib':24,'ram_gib':192})
        self.assertEqual((p.chunk,p.ring_mib,p.wave_experts),(2048,512,8))
    def test_short_append_keeps_actual_length(self):
        p=seed_for_recipe({'model':'exl3-3','vram_gib':24,'ram_gib':128},append_tokens=17)
        self.assertEqual(p.chunk,17);self.assertEqual(p.expert_row_tile,17)
    def test_q2_not_relabelled(self):
        with self.assertRaises(ContractError):seed_for_recipe({'model':'q2','vram_gib':24,'ram_gib':128})
    def test_peak_is_max_of_phases(self):
        w=workspace_budget(Inputs());self.assertEqual(w['peak_bytes'],max(w['phase_total_bytes'].values()))
        self.assertLess(w['peak_bytes'],sum(w['phase_total_bytes'].values()))
    def test_attention_slab_bounded(self):
        a=workspace_budget(Inputs(chunk=1024));b=workspace_budget(Inputs(chunk=4096))
        self.assertEqual(a['phase_terms']['attention'],b['phase_terms']['attention'])
    def test_expert_wave_bounded(self):
        a=workspace_budget(Inputs(chunk=1024));b=workspace_budget(Inputs(chunk=4096))
        self.assertEqual(a['phase_terms']['moe']['gate_up_hadamard_inputs_fp16'],b['phase_terms']['moe']['gate_up_hadamard_inputs_fp16'])
    def test_contribution_scales_with_chunk(self):
        a=workspace_budget(Inputs(chunk=1024));b=workspace_budget(Inputs(chunk=2048))
        self.assertEqual(2*a['phase_terms']['moe']['contributions_by_token_topk_fp32'],b['phase_terms']['moe']['contributions_by_token_topk_fp32'])
    def test_reconstruct_is_explicit(self):
        a=workspace_budget(Inputs());b=workspace_budget(Inputs(reconstruction='one_projection_with_hadamard_temp'))
        self.assertEqual(b['peak_bytes']-a['peak_bytes'],47185920)
    def test_plan_does_not_promote_to_gpu(self):
        p=make_plan(Inputs());self.assertTrue(p['fit_under_declared_design_contract']);self.assertFalse(p['can_launch'])
        self.assertIsNone(p['performance_prediction'])
    def test_device_oom(self):
        p=make_plan(Inputs(mandatory_device_bytes=24*GiB));self.assertFalse(p['fit_under_declared_design_contract'])
    def test_host_oom(self):
        p=make_plan(Inputs(ram_gib=16));self.assertFalse(p['fit_under_declared_design_contract'])
    def test_ring_bad(self):
        p=make_plan(Inputs(ring_mib=1));self.assertFalse(p['fit_under_declared_design_contract'])
    def test_no_all_model_host_copy(self):
        p=make_plan(Inputs(ram_gib=128));self.assertLess(p['host']['warm_cache_bytes'],128*GiB)
        self.assertGreater(p['cold_expert_count_design'],0)
    def test_256_full_expert_storage_possible_in_design(self):
        p=make_plan(Inputs());self.assertEqual(p['cold_expert_count_design'],0)
    def test_more_slots_reduce_gpu_cache(self):
        p=make_plan(Inputs(context=131072,slots=1));q=make_plan(Inputs(context=131072,slots=5))
        self.assertLess(q['device']['hot_slots'],p['device']['hot_slots'])
    def test_pinned_io_not_whole_bank(self):
        p=make_plan(Inputs());self.assertEqual(p['host']['noncache_terms']['pinned_expert_staging'],2*p['ring']['allocated_bytes'])
    def test_no_graph_claim(self):
        with self.assertRaises(ContractError):Inputs(graph_capture=True)
    def test_invalid_context(self):
        for kw in ({'context':1048577},{'context':32,'chunk':64},{'slots':0},{'chunk':True},{'reconstruction':'all_experts'}):
            with self.assertRaises(ContractError):Inputs(**kw)
    def test_candidates_no_winner_no_monotonic_assumption(self):
        plans=candidate_plans(Inputs(),chunks=(256,1024,3072),rings=(256,512))
        self.assertEqual(len(plans),6);self.assertTrue(all(p['performance_prediction'] is None for p in plans))
    def test_inputs_digest_changes(self):
        self.assertNotEqual(make_plan(Inputs())['plan_digest'],make_plan(Inputs(chunk=1024))['plan_digest'])
    def test_hot_and_warm_exclusive_counts(self):
        for ram in (128,192,256,384):
            p=make_plan(Inputs(ram_gib=ram))
            self.assertEqual(p['device']['hot_slots']+p['host']['warm_slots']+p['cold_expert_count_design'],15360)
    def test_no_ratio_double_divide(self):
        self.assertEqual(state_budget(8192)['source_banks'][0]['logical_rows'],4096)

class RouteTests(unittest.TestCase):
    def setUp(self):self.g=Geometry(hidden=128,intermediate=128,layers=1,experts=8,top_k=2)
    def test_stable_group_order(self):
        g=group_routes([[3,1],[1,3]],[[.2,.8],[.7,.3]],self.g)
        self.assertEqual(list(g),[1,3]);self.assertEqual([(x.token,x.slot) for x in g[1]],[(0,1),(1,0)])
    def test_duplicate_rejected(self):
        with self.assertRaises(ContractError):group_routes([[1,1]],[[1,1]],self.g)
    def test_out_of_bounds(self):
        with self.assertRaises(ContractError):group_routes([[1,8]],[[1,1]],self.g)
    def test_missing_width(self):
        with self.assertRaises(ContractError):group_routes([[1]],[[1]],self.g)
    def test_nan_weight(self):
        with self.assertRaises(ContractError):group_routes([[1,2]],[[float('nan'),1]],self.g)
    def test_one_fetch_for_many_segments(self):
        g=group_routes([[1,2]]*512,[[.5,.5]]*512,self.g)
        p=make_waves(g,2,16,g=self.g)
        self.assertEqual(p['stats']['expert_loads'],2);self.assertEqual(p['stats']['segments'],64)
    def test_gpu_hit_vs_ram_hit(self):
        g=group_routes([[1,2],[3,4]],[[.5,.5]]*2,self.g)
        p=make_waves(g,2,16,gpu_ready=[1],ram_ready=[1,2,3],g=self.g);size=bundle_layout(self.g)['source_payload_bytes']
        self.assertEqual(p['stats']['h2d_payload_bytes'],3*size);self.assertEqual(p['stats']['ssd_logical_bundle_bytes'],size)
    def test_ram_resident_not_zero_h2d(self):
        g=group_routes([[1,2]],[[1,1]],self.g)
        p=make_waves(g,2,16,ram_ready=[1,2],g=self.g)
        self.assertEqual(p['stats']['ssd_logical_bundle_bytes'],0);self.assertGreater(p['stats']['h2d_payload_bytes'],0)
    def test_unselected_experts_not_loaded(self):
        g=group_routes([[1,2]],[[1,1]],self.g);p=make_waves(g,2,16,g=self.g)
        self.assertEqual([j['expert'] for w in p['waves'] for j in w],[1,2])
    def test_executor_bound_and_output(self):
        opened=[];peak=[0];calls=[]
        @contextlib.contextmanager
        def provider(e):
            opened.append(e);calls.append(e);peak[0]=max(peak[0],len(opened))
            try:yield e
            finally:opened.remove(e)
        def compute(e,assign):return [[(a.token+1)*(e+1)*a.weight] for a in assign]
        ids=[[1,2],[3,1],[2,5]];ws=[[.4,.6]]*3
        result,stats=execute_reference(ids,ws,provider,compute,g=self.g,wave_experts=2,row_tile=1)
        expected=[[sum((t+1)*(e+1)*w for e,w in zip(row,ws[t]))] for t,row in enumerate(ids)]
        self.assertEqual(result,expected);self.assertLessEqual(peak[0],2);self.assertEqual(len(calls),4);self.assertEqual(opened,[])
    def test_executor_missing_expert_error(self):
        @contextlib.contextmanager
        def provider(e):yield None
        with self.assertRaises(ContractError):execute_reference([[1,2]],[[1,1]],provider,lambda *_:[],g=self.g)
    def test_executor_cancel_releases_handles(self):
        open_handles=[]
        @contextlib.contextmanager
        def provider(e):
            open_handles.append(e)
            try:yield e
            finally:open_handles.remove(e)
        def compute(*_):raise RuntimeError('cancel')
        with self.assertRaises(RuntimeError):execute_reference([[1,2]],[[1,1]],provider,compute,g=self.g)
        self.assertEqual(open_handles,[])
    def test_wave_and_tile_do_not_change_reference(self):
        rng=random.Random(12);ids=[rng.sample(range(8),2) for _ in range(39)];w=[[.4,.6]]*len(ids)
        @contextlib.contextmanager
        def provider(e):yield e
        def compute(e,aa):return [[(a.token+3)*(.125*e)*a.weight] for a in aa]
        baseline=None
        for wave in (1,2,4):
            for tile in (1,7,64):
                actual,_=execute_reference(ids,w,provider,compute,g=self.g,wave_experts=wave,row_tile=tile)
                if baseline is None:baseline=actual
                self.assertEqual(actual,baseline)
    def test_empty_routes(self):
        self.assertEqual(group_routes([],[],self.g),{})

# Synthetic safetensors with REAL TP1 shapes; sparse file for payload, actual marker bytes.
# Deliberately NOT advertised as real pretrained model data.
def write_fixture(path, *, expert=0,layer=0,bad_marker=False,drop=None,wrong_shape=False,keep_projections=None,scalar_marker=False):
    hdr={};offset=0;markers=[]
    for c in components():
        key=f'layers.{layer}.ffn.experts.{expert}.{c.projection}.{c.suffix}'
        if drop==f'{c.projection}.{c.suffix}':continue
        if keep_projections is not None and c.projection not in keep_projections:continue
        shape=[] if scalar_marker and c.suffix=='mul1' else list(c.shape)
        if wrong_shape and c.suffix=='trellis':shape=[len(shape),c.payload_bytes//6]
        hdr[key]={'dtype':c.dtype,'shape':shape,'data_offsets':[offset,offset+c.payload_bytes]}
        if c.suffix=='mul1':markers.append(offset)
        offset+=c.payload_bytes
    raw=json.dumps(hdr,separators=(',',':')).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw);f.truncate(8+len(raw)+offset)
        for off in markers:f.seek(8+len(raw)+off);f.write(struct.pack('<I',0 if bad_marker else MUL1))

class AuditTests(unittest.TestCase):
    def test_actual_shape_sparse_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.safetensors';write_fixture(p)
            result=audit_exl3(d,['x.safetensors'])
            self.assertEqual(result['routed_payload_bytes'],13315596);self.assertFalse(result['payload_sha256_verified'])
            self.assertEqual(len(result['bundles'][0]['components']),12)
    def test_full_coverage_not_faked(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors')
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors'],complete=True)
    def test_bad_marker(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors',bad_marker=True)
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors'])
    def test_missing_component(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors',drop='w3.svh')
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors'])
    def test_wrong_layout(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors',wrong_shape=True)
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors'])
    def test_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors',expert=384)
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors'])
    def test_duplicate_file(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors')
            with self.assertRaises(ContractError):audit_exl3(d,['x.safetensors','x.safetensors'])
    def test_no_directory_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ContractError):audit_exl3(d,['../x.safetensors'])
    def test_multishard_unique_bundle_ids(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors');write_fixture(Path(d)/'y.safetensors',expert=1)
            r=audit_exl3(d,['x.safetensors','y.safetensors']);self.assertEqual(r['expert_count'],2)
    def test_components_split_across_files(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'a.safetensors',keep_projections=['w1','w3'])
            write_fixture(Path(d)/'b.safetensors',keep_projections=['w2'])
            r=audit_exl3(d,['a.safetensors','b.safetensors'])
            self.assertEqual(r['expert_count'],1)
            self.assertEqual(r['routed_payload_bytes'],13315596)
            self.assertEqual({c['file'] for c in r['bundles'][0]['components']},{'a.safetensors','b.safetensors'})
    def test_scalar_marker_layout(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'a.safetensors',scalar_marker=True)
            r=audit_exl3(d,['a.safetensors'])
            self.assertEqual(r['expert_count'],1)
    def test_marker_check_can_be_separately_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d)/'x.safetensors',bad_marker=True)
            r=audit_exl3(d,['x.safetensors'],read_markers=False)
            self.assertIn('UNCHECKED',r['status'])

def measured_records():
    # Unit-test fabricated measurements: never exported as actual run evidence.
    out=[]
    for name,prefill,decode in [('a',100,20),('b',90,60)]:
        for i in range(3):out.append(dict(evidence='MODEL_PERFORMANCE_MEASURED',correctness_passed=True,peak_within_budget=True,
            model_revision='m',quant_digest='q',engine_commit='e',hardware_digest='h',workload_digest='w',kv_format='bf16',
            configured_slots=1,math_policy='fixed',warmth_policy='warm-new-prefix',trial_id=f'{name}-{i}',candidate_id=name,
            prefill_ms=prefill+i,refill_ms=10,following_decode_ms=decode,max_decode_gap_ms=20,
            ssd_physical_bytes=10,h2d_payload_bytes=20,gpu_peak_bytes=30,host_peak_bytes=40))
    return out
class TuneTests(unittest.TestCase):
    def test_winner_accounts_refill_decode_not_prefill_only(self):
        r=choose_measured(measured_records(),max_decode_gap_ms=30);self.assertEqual(r['winner'],'a')
    def test_qos_can_reject_all(self):
        self.assertIsNone(choose_measured(measured_records(),max_decode_gap_ms=10)['winner'])
    def test_synthetic_rejected(self):
        r=measured_records();r[0]['evidence']='SYNTHETIC_TEST'
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)
    def test_identity_mismatch(self):
        r=measured_records();r[-1]['hardware_digest']='other'
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)
    def test_single_repeat_not_enough(self):
        with self.assertRaises(ContractError):choose_measured(measured_records()[:1],max_decode_gap_ms=30)
    def test_duplicate_trial_not_repeat(self):
        r=measured_records();r[-1]['trial_id']=r[0]['trial_id']
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)
    def test_quality_failure_not_fast_winner(self):
        r=measured_records();r[0]['correctness_passed']=False
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)
    def test_unknown_traffic_not_zero(self):
        r=measured_records();r[0]['ssd_physical_bytes']=None
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)
    def test_nonfinite_latency(self):
        r=measured_records();r[0]['prefill_ms']=float('nan')
        with self.assertRaises(ContractError):choose_measured(r,max_decode_gap_ms=30)

if __name__=='__main__':unittest.main()
