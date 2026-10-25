"""Analytical PPO/route checks independent of cluster packing and rollouts."""
import importlib.util
import math
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'install elecload[train]')
class PairedObjectiveTests(unittest.TestCase):
    def data(self, torch, rows=2, width=1):
        return {'old_log_probs': torch.zeros(rows,width,dtype=torch.float64),
                'generated_mask': torch.ones(rows,width,dtype=torch.float64),
                'response_mask': torch.ones(rows,width,dtype=torch.float64),
                'advantages': torch.ones(rows,width,dtype=torch.float64),
                'row_kind': torch.zeros(rows,dtype=torch.long),
                'distributed_padding': torch.zeros(rows,dtype=torch.bool),
                'task_weight': torch.ones(rows,dtype=torch.float64),
                'contract_weight': torch.zeros(rows,dtype=torch.float64),
                'kl_weight': torch.zeros(rows,dtype=torch.float64)}

    def test_clipping_blocks_gradients_beyond_both_improvement_bounds(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        data = self.data(torch)
        data['advantages'][1] = -1
        new = torch.tensor([[math.log(1.3)],[math.log(.7)]],dtype=torch.float64,requires_grad=True)
        loss, _, _ = credit_objective(new,data)
        self.assertAlmostEqual(float(loss.detach()), -.4)
        loss.backward()
        torch.testing.assert_close(new.grad,torch.zeros_like(new))

    def test_unclipped_gradient_and_frozen_labels(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        data = self.data(torch)
        data['advantages'][1] = -1
        data['old_log_probs'].requires_grad_();data['advantages'].requires_grad_()
        new = torch.tensor([[math.log(1.1)],[math.log(.9)]],dtype=torch.float64,requires_grad=True)
        loss, _, _ = credit_objective(new,data)
        loss.backward()
        torch.testing.assert_close(new.grad,torch.tensor([[-1.1],[.9]],dtype=torch.float64))
        self.assertIsNone(data['old_log_probs'].grad)
        self.assertIsNone(data['advantages'].grad)

    def test_microbatch_partition_preserves_global_weights(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        data=self.data(torch,rows=3,width=2)
        data['task_weight']=torch.tensor([.1,.2,.7],dtype=torch.float64)
        a=torch.full((3,2),-.1,dtype=torch.float64,requires_grad=True)
        full,_,_=credit_objective(a,data);full.backward()
        b=a.detach().clone().requires_grad_()
        parts=[credit_objective(b[i:i+1],{k:v[i:i+1] for k,v in data.items()})[0] for i in range(3)]
        summed=sum(parts);summed.backward()
        torch.testing.assert_close(summed,full)
        torch.testing.assert_close(b.grad,a.grad)

    def test_natural_contract_mask_and_version_are_not_inferred(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        version='paired-main-pre4-natural-v6'
        data=self.data(torch,rows=1,width=2)
        new=torch.zeros(1,2,dtype=torch.float64,requires_grad=True)
        with self.assertRaisesRegex(ValueError,'explicit contract mask'):
            credit_objective(new,data,objective_version=version)
        data['task_weight'].zero_();data['contract_weight'].fill_(.5)
        data['contract_mask']=torch.tensor([[1.,0.]],dtype=torch.float64)
        loss,_,_=credit_objective(new,data,objective_version=version);loss.backward()
        torch.testing.assert_close(new.grad,torch.tensor([[.5,0.]],dtype=torch.float64))
        data['contract_mask'][0,0]=2
        with self.assertRaisesRegex(ValueError,'binary subset'):
            credit_objective(new,data,objective_version=version)
        with self.assertRaises(ValueError):
            credit_objective(new,data,objective_version='unknown')

    def test_paired_route_loss_moves_probability_toward_better_branch(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        data=self.data(torch)
        data['generated_mask'].zero_();data['task_weight'].zero_();data['row_kind'].fill_(1)
        data.update(q_pair_index=torch.tensor([0,0]),q_candidate=torch.tensor([0,1]),
            route_valid=torch.ones(2),route_scoreable=torch.ones(2),route_delta=torch.full((2,),2.),
            route_M=torch.full((2,),4.),q_reference_score=torch.zeros(2))
        scores=torch.nn.Parameter(torch.tensor([[-1.],[-1.]],dtype=torch.float64))
        optimizer=torch.optim.SGD([scores],lr=.1)
        loss,metrics,_=credit_objective(scores,data,objective_version='paired-logistic-post-v2')
        self.assertAlmostEqual(float(metrics['paired_credit/route']),-.25*math.log(.5))
        loss.backward();optimizer.step()
        self.assertGreater(float(torch.softmax(scores.flatten(),0)[1]),.5)
        with self.assertRaisesRegex(ValueError,'split or duplicated'):
            credit_objective(scores[:1],{k:v[:1] for k,v in data.items()},objective_version='paired-logistic-post-v2')

    def test_nonfinite_actual_tokens_fail(self):
        import torch
        from castfactory.training.paired_credit_loss import credit_objective
        data=self.data(torch)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            credit_objective(torch.full((2,1),float('nan')),data)
