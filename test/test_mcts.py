"""Regression tests for reward propagation and bounded experiment execution."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from rdkit import Chem
from Generator import mcts as fixed
from Generator.search_budget import BudgetedReward, SearchBudgetExceeded, install_request_timeout


class ConstantReward:
    vmin = -100
    def __init__(self, value=0):
        self.value = value
        self.calls = 0
    def reward(self, mol):
        self.calls += 1
        return self.value


def fake_fine(rows, interrupted=False):
    class Fine:
        def __init__(self, **kwargs):
            self.init_score = 0
            self.max_score = max([r for _, r in rows], default=-1e5)
            self.generated_smiles = {"SMILES": [s for s, _ in rows],
                "Reward": [r for _, r in rows], "Step": [1] * len(rows)}
            self.completed_steps = 0 if interrupted else 1
        def search(self, **kwargs):
            if interrupted:
                raise SearchBudgetExceeded()
    return Fine


class FixedMCTSTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        os.environ.pop('GOBLET_SEARCH_DEADLINE_EPOCH', None)
    def tearDown(self):
        self.env.stop()

    def coarse_case(self, rows, interrupted=False):
        search = fixed.CoarseMCTS('C', ConstantReward(), None)
        search.root.visit = 1
        node = fixed.MolNode('C')
        search.root.add_Node(node)
        search.current_node = node
        with patch.object(fixed, 'FineMCTS', fake_fine(rows, interrupted)):
            reward = search._simulate(1)
        search._backprop(reward)
        return search, node

    def test_coarse_rewards_change_exploitation(self):
        _, low = self.coarse_case([('CC', 0.2)])
        _, high = self.coarse_case([('CC', 0.8)])
        self.assertAlmostEqual(low.cum_score, 0.2/1.2)
        self.assertAlmostEqual(high.cum_score, 0.8/1.8)
        self.assertGreater(high.calc_UCB(), low.calc_UCB())
        self.assertEqual(low.visit, 1)

    def test_coarse_maximum_and_no_nan_children(self):
        search, node = self.coarse_case([('CC', 0.2), ('CCC', 0.8)])
        self.assertEqual({n.smiles for n in node.children}, {'CC', 'CCC'})
        self.assertAlmostEqual(node.imm_score, 0.8/1.8)
        self.assertEqual(search.coarse_updates[-1]['raw_reward'], 0.8)
        for child in node.children:
            self.assertTrue(all(g.depth == child.depth + 1 for g in child.children))

    def test_no_candidate_keeps_failure_penalty(self):
        _, node = self.coarse_case([])
        self.assertAlmostEqual(node.cum_score, -10000/10001)
        self.assertAlmostEqual(node.imm_score, -10000/10001)

    def test_interruption_without_results_is_not_failure(self):
        search, node = self.coarse_case([], interrupted=True)
        self.assertEqual(search.stop_reason, 'walltime_budget')
        self.assertEqual(node.visit, 0)

    def test_interruption_retains_valid_observations(self):
        search, node = self.coarse_case([('CC', 0.8)], interrupted=True)
        self.assertEqual(len(search.generated_smiles), 1)
        self.assertFalse(search.generated_smiles.iloc[0]['FineStepComplete'])
        self.assertAlmostEqual(node.cum_score, 0.8/1.8)

    def test_elite_not_first_row_and_limit(self):
        search = fixed.CoarseMCTS('C', ConstantReward(), None)
        df = pd.DataFrame({'SMILES': ['C','CC','CCC','O','N',None],
                           'Reward': [0.1,0.2,0.3,0.9,0.4,100]})
        result = search._select_next_state(df, 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['SMILES'], 'O')
        self.assertTrue(result['SMILES'].map(lambda s: isinstance(s,str)).all())

    def test_removal_nodes_receive_subsequent_reward(self):
        search = fixed.CoarseMCTS('CCO', ConstantReward(), None)
        search.set_state(search.root)
        node = next(n for n in search.root.children if n.smiles == 'CC')
        search.current_node = node
        with patch.object(fixed, 'FineMCTS', fake_fine([('CCC',0.8)])):
            reward = search._simulate(1)
        search._backprop(reward)
        self.assertAlmostEqual(node.cum_score, 0.8/1.8)
        self.assertAlmostEqual(search.root.cum_score, node.cum_score)

    def test_depth_pruning_does_not_raise_or_loop(self):
        for node_class in (fixed.AtomNode,fixed.MolNode):
            parent = node_class('C')
            child = node_class('CC')
            grandchild = node_class('CCC')
            child.add_Node(grandchild)
            parent.add_Node(child)
            self.assertEqual(grandchild.depth,2)
            child.remove_Node()
            self.assertEqual(parent.children,[])
        search = fixed.CoarseMCTS('C',ConstantReward(),None)
        search.root.add_Node(fixed.MolNode('CC'))
        search.max_seq = 1
        search._select()
        self.assertIs(search.current_node,search.root)
        self.assertEqual(search.root.children,[])

    def test_fine_simulate_and_backup_normalize_once(self):
        class Sampler:
            def sample(self, **kwargs): return ['CC']
        for reward in (0.2,0.8,-0.2):
            search = fixed.FineMCTS('C',ConstantReward(reward),Sampler())
            node = fixed.AtomNode('C',att_mid=1)
            child = fixed.AtomNode('CC',att_mid=1)
            child.visit = 1
            node.add_Node(child)
            search.root.add_Node(node)
            search.current_node = node
            search._simulate()
            search._backprop()
            self.assertAlmostEqual(child.cum_score,reward/(1+abs(reward)))
            self.assertAlmostEqual(node.cum_score,child.cum_score)

    def test_deadline_stops_even_cached_reward(self):
        delegate=ConstantReward(0.8)
        wrapper=BudgetedReward(delegate)
        wrapper.reward(Chem.MolFromSmiles('C'))
        os.environ['GOBLET_SEARCH_DEADLINE_EPOCH']=str(time.time()-1)
        with self.assertRaises(SearchBudgetExceeded):
            wrapper.reward(Chem.MolFromSmiles('C'))

    def test_cache_key_ignores_atom_maps(self):
        delegate=ConstantReward(0.8)
        with tempfile.TemporaryDirectory() as directory:
            journal=Path(directory)/'rewards.jsonl'
            wrapper=BudgetedReward(delegate,journal)
            wrapper.reward(Chem.MolFromSmiles('C'))
            wrapper.reward(Chem.MolFromSmiles('[CH4:1]'))
            self.assertEqual(delegate.calls,1)
            self.assertEqual(wrapper.cache_hits,1)
            self.assertEqual(len(journal.read_text().splitlines()),2)

    def test_failed_rewards_are_not_cached(self):
        delegate=ConstantReward(-100)
        wrapper=BudgetedReward(delegate)
        for _ in range(2): wrapper.reward(Chem.MolFromSmiles('C'))
        self.assertEqual(delegate.calls,2)

    def test_http_timeout_is_bounded(self):
        import requests
        with patch.object(requests,'post') as original:
            install_request_timeout(600)
            try:
                requests.post('http://example.invalid',timeout=900)
                self.assertEqual(original.call_args.kwargs['timeout'],(10,600))
            finally:
                requests.post=original

    def test_non_multiple_step_budget_is_exact(self):
        search=fixed.CoarseMCTS('C',ConstantReward(),None)
        def simulate(n_step):
            search.completed_steps += n_step
            return -10000.0
        with patch.object(search,'_simulate',side_effect=simulate):
            search.search(23,20)
        self.assertEqual(search.completed_steps,23)

    def test_nonpositive_fine_batch_is_rejected(self):
        for n_step_fine in (0, -1):
            search = fixed.CoarseMCTS('C', ConstantReward(), None)
            with self.assertRaisesRegex(ValueError, 'n_step_fine must be positive'):
                search.search(3, n_step_fine)


if __name__ == '__main__':
    unittest.main(verbosity=2)
