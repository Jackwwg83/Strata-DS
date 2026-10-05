from __future__ import annotations
import copy
import unittest
from strata_ds_lab.common import ContractError, read_json
from strata_ds_lab.upgrade0139 import (
    ROOT, ring_capacity, expert_file_io_advice, phase_peak,
    require_trial_slots, merged_task_order, validate_profiles, validate_design, planned_matrix,
)

class RingBudgetTests(unittest.TestCase):
    def test_floor_is_not_slot_roundup(self):
        r=ring_capacity(1024,300)
        self.assertEqual(r['ring_slots'],3)
        self.assertEqual(r['ring_allocated_bytes'],900)
        self.assertEqual(r['unspent_budget_bytes'],124)
        self.assertFalse(r['can_launch'])
    def test_larger_bundle_means_fewer_slots(self):
        self.assertGreater(ring_capacity(4096,128)['ring_slots'],ring_capacity(4096,512)['ring_slots'])
    def test_zero_budget_fails(self):
        with self.assertRaises(ContractError):ring_capacity(0,64)
    def test_minimum_cannot_force_overspend(self):
        with self.assertRaises(ContractError):ring_capacity(127,64,2)
    def test_exact_minimum_fits(self):
        self.assertEqual(ring_capacity(128,64,2)['ring_slots'],2)
    def test_cap(self):
        self.assertEqual(ring_capacity(2048,64,maximum_slots=8)['ring_slots'],8)
    def test_invalid_cap(self):
        with self.assertRaises(ContractError):ring_capacity(2048,64,4,3)
    def test_zero_stride(self):
        with self.assertRaises(ContractError):ring_capacity(1024,0)
    def test_bool_is_not_bytes(self):
        with self.assertRaises(ContractError):ring_capacity(True,64)
    def test_unknown_stride(self):
        with self.assertRaises(ContractError):ring_capacity(1024,None)
    def test_large_integer_no_float_loss(self):
        n=(1<<63)+12345
        r=ring_capacity(n,77)
        self.assertEqual(r['ring_allocated_bytes']+r['unspent_budget_bytes'],n)

class IOAdviceTests(unittest.TestCase):
    def advice(self,a=100,f=20,w=60,d=False):
        return expert_file_io_advice(available_after_residency_bytes=a,future_reserve_bytes=f,
                                    file_expert_working_set_bytes=w,direct_io_qualified=d)
    def test_post_residency_not_double_subtracted(self):
        self.assertEqual(self.advice()['advisory_room_bytes'],80)
        self.assertEqual(self.advice()['candidate'],'BUFFERED_AB_CANDIDATE')
    def test_exact_file_working_set_fits(self):
        self.assertEqual(self.advice(w=80)['candidate'],'BUFFERED_AB_CANDIDATE')
    def test_direct_only_if_externally_qualified(self):
        self.assertEqual(self.advice(w=81,d=True)['candidate'],'DIRECT_AB_CANDIDATE')
    def test_no_direct_assumption(self):
        self.assertEqual(self.advice(w=81)['candidate'],'BOUNDED_BUFFERED_PRESSURE_TEST_REQUIRED')
    def test_no_reads(self):
        self.assertEqual(self.advice(w=0)['candidate'],'NO_EXPERT_FILE_READS_IN_THIS_PHASE')
    def test_saturates(self):
        self.assertEqual(self.advice(a=3,f=10)['advisory_room_bytes'],0)
    def test_never_qualifies_launch(self):
        self.assertFalse(self.advice()['can_launch'])
    def test_unknown_ram_rejected(self):
        with self.assertRaises(ContractError):self.advice(a=None)
    def test_bool_size_rejected(self):
        with self.assertRaises(ContractError):self.advice(w=False)
    def test_unknown_direct_flag_rejected(self):
        with self.assertRaises(ContractError):self.advice(d='auto')
    def test_negative_rejected(self):
        with self.assertRaises(ContractError):self.advice(f=-1)

class PhaseAndSlotTests(unittest.TestCase):
    def phase(self,slots=1,terms=None,phase='decode',per=100):
        return phase_peak(phase=phase,capacity_bytes=1000,allocated_slots=slots,per_slot_bytes=per,
                          co_live_shared_bytes=terms if terms is not None else {'mandatory':300,'ring':200})
    def test_all_allocated_slots_count(self):
        self.assertEqual(self.phase(slots=5)['supplied_peak_bytes'],1000)
    def test_excess(self):
        r=self.phase(slots=6)
        self.assertFalse(r['arithmetic_fits']);self.assertEqual(r['excess_bytes'],100)
    def test_arithmetic_fit_is_not_launch(self):
        self.assertTrue(self.phase()['arithmetic_fits']);self.assertFalse(self.phase()['can_launch'])
    def test_unknown_term_rejected(self):
        with self.assertRaises(ContractError):self.phase(terms={'mandatory':None})
    def test_empty_terms_rejected(self):
        with self.assertRaises(ContractError):self.phase(terms={})
    def test_non_byte_float_rejected(self):
        with self.assertRaises(ContractError):self.phase(terms={'mandatory':1.5})
    def test_unknown_phase_rejected(self):
        with self.assertRaises(ContractError):self.phase(phase='all_phases_added')
    def test_bool_slots_rejected(self):
        with self.assertRaises(ContractError):self.phase(slots=True)
    def test_negative_slot_bytes_rejected(self):
        with self.assertRaises(ContractError):self.phase(per=-1)
    def test_requested_effective_match(self):
        require_trial_slots(5,5)
    def test_downclamp_not_success(self):
        with self.assertRaisesRegex(ContractError,'CAPACITY_MISMATCH'):require_trial_slots(5,1)
    def test_unknown_effective_rejected(self):
        with self.assertRaises(ContractError):require_trial_slots(5,None)

class DependencyTests(unittest.TestCase):
    def tasks(self):return [{'id':'SD-A','depends_on':[]},{'id':'SD-B','depends_on':['SD-A']}]
    def test_merge_order(self):
        order=merged_task_order(self.tasks(),[{'id':'U-A','depends_on':[]}],{'SD-A':['U-A']})
        self.assertEqual(order,['U-A','SD-A','SD-B'])
    def test_cycle_rejected(self):
        with self.assertRaises(ContractError):merged_task_order(self.tasks(),[],{'SD-A':['SD-B']})
    def test_unknown_dependency(self):
        with self.assertRaises(ContractError):merged_task_order(self.tasks(),[],{'SD-A':['MISSING']})
    def test_duplicate_id(self):
        with self.assertRaises(ContractError):merged_task_order(self.tasks(),[{'id':'SD-A','depends_on':[]}],{})
    def test_bad_augmentation(self):
        with self.assertRaises(ContractError):merged_task_order(self.tasks(),[],{'MISSING':[]})

class PlanContractTests(unittest.TestCase):
    def cfg(self):return read_json(ROOT/'specs/upstream-v0139.profiles.json')
    def test_all_baseline_files_and_pins_preserved(self):
        r=validate_design();self.assertEqual(r['preserved_recipe_count'],11)
        self.assertEqual(r['merged_task_count'],30);self.assertFalse(r['can_launch'])
    def test_eight_profiles(self):
        self.assertEqual(len(validate_profiles(self.cfg())),8)
    def test_idle_slot_profile_has_one_client(self):
        d={p['id']:p for p in self.cfg()['profiles']}
        self.assertEqual((d['slots5-idle']['requested_slots'],d['slots5-idle']['offered_concurrency']),(5,1))
    def test_queue_profile_exceeds_slots(self):
        d={p['id']:p for p in self.cfg()['profiles']}
        self.assertEqual((d['slots2-queue']['requested_slots'],d['slots2-queue']['offered_concurrency']),(2,5))
    def test_guessed_ds_state_rejected(self):
        c=self.cfg();c['ds_session_bytes']=int(.56*(1<<30))
        with self.assertRaises(ContractError):validate_profiles(c)
    def test_enable_speculation_rejected(self):
        c=self.cfg();c['profiles'][0]['speculation']=True
        with self.assertRaises(ContractError):validate_profiles(c)
    def test_enable_launch_rejected(self):
        c=self.cfg();c['profiles'][0]['can_launch']=True
        with self.assertRaises(ContractError):validate_profiles(c)
    def test_duplicate_profile_rejected(self):
        c=self.cfg();c['profiles'][1]['id']=c['profiles'][0]['id']
        with self.assertRaises(ContractError):validate_profiles(c)
    def test_initial_matrix_size(self):
        rows=planned_matrix(ROOT,'exl3-3-r256-v24',True)
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['effective_slots'] is None and not r['can_launch'] for r in rows))
    def test_full_profile_matrix(self):
        rows=planned_matrix(ROOT,'exl3-3-r256-v24')
        self.assertEqual(len(rows),48)
        self.assertEqual(len({r['plan_id'] for r in rows}),48)
    def test_stable_matrix(self):
        self.assertEqual(planned_matrix(ROOT,'q2-r128-v24',True,1),planned_matrix(ROOT,'q2-r128-v24',True,1))
    def test_recipe_traversal_rejected(self):
        with self.assertRaises(ContractError):planned_matrix(ROOT,'../../secret')
    def test_excess_repeat_rejected(self):
        with self.assertRaises(ContractError):planned_matrix(ROOT,'exl3-3-r256-v24',repeats=21)

if __name__=='__main__':unittest.main()
