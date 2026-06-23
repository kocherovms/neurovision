from collections import defaultdict
import math
import numpy as np
import torch
import torch.optim

class LrSchedulerWrapper:
    def __init__(self, optimizer, hp_learn_rate_params, epochs_count):
        if hp_learn_rate_params.plateau is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **hp_learn_rate_params.plateau._asdict())
            self.step = self.step_plateau
        elif hp_learn_rate_params.linear is not None:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **hp_learn_rate_params.linear._asdict(), total_iters=epochs_count)
            self.step = self.step_linear
        else:
            self.scheduler = None
            self.step = self.step_dummy

    def step_dummy(self, value):
        pass
    
    def step_plateau(self, value):
        self.scheduler.step(value)

    def step_linear(self, value=None):
        self.scheduler.step()

class ModelModeContextManager:
    def __init__(self, model, target_mode):
        self.training = model.training
        self.model = model

        match target_mode:
            case 'eval': self.model.eval()
            case 'train': self.model.train()
            case _: assert False, f'Unsupported {target_mode=}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.training:
            self.model.train()
        else:
            self.model.eval()
            
        return False

def eval_guard(model):
    return ModelModeContextManager(model, 'eval')

def train_guard(model):
    return ModelModeContextManager(model, 'train')

def numpy_dtype_to_torch_dtype(np_dtype):
    return torch.from_numpy(np.array([], dtype=np_dtype)).dtype

def get_grad_norm_groups(model, group_names):
    norms = defaultdict(list)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_sum_squares = torch.sum(param.grad.detach() ** 2).item()

            for group_name in group_names:
                if group_name in name:
                    norms[group_name].append(grad_sum_squares)
                    break

    result = {}
    
    for k, v in norms.items():
        result[k] = math.sqrt(sum(v)) if v else 0.0

    return result
    
def get_old_linear_anneal(start_value, end_value, steps_count):
    def anneal_param(step):
        if steps_count <= 0 or step >= steps_count:
            return end_value
            
        frac = 1 - step / steps_count
        return (start_value - end_value) * frac + end_value

    return anneal_param

def get_anneal(anneal_name, *args, **kwargs):
    match anneal_name:
        case 'const': return get_const_anneal(*args, **kwargs)
        case 'linear': return get_linear_anneal(*args, **kwargs)
        case 'trapezoid': return get_trapezoid_anneal(*args, **kwargs)
        case 'shark_fin': return get_shark_fin_anneal(*args, **kwargs)
        case _: assert False, f'Unsupported {anneal_name=}'

def get_const_anneal(a):
    def thunk(t):
        return a

    return thunk

def get_linear_anneal(a, b):
    def thunk(t):
        assert 0 <= t <= 1, t
        return a + t * (b - a)

    return thunk

def get_trapezoid_anneal(a, b, c, a_dur, b_dur):
    assert a_dur >= 0, a_dur
    assert b_dur >= 0, b_dur
    assert 0 <= (a_dur + b_dur) <= 1, (a_dur, b_dur)
    c_dur = 1 - a_dur - b_dur
    
    def thunk(t):
        if t < a_dur and a_dur > 0:
            return a + t * (b - a) / a_dur
        elif a_dur <= t < (a_dur + b_dur):
            return b
        else:
            if c_dur > 0:
                t = t - a_dur - b_dur
                return b + t * (c - b) / c_dur
            else:
                return c

    return thunk

def get_shark_fin_anneal(a, b, warm_dur=0.05):
    assert 0 <= warm_dur <= 1, warm_dur
    anneal_dur = 1 - warm_dur
    assert anneal_dur >= 0, anneal_dur
    
    def thunk(t):
        if t < warm_dur and warm_dur > 0:
            return 0 + t * (a - 0) / warm_dur
            
        progress = (t - warm_dur) / anneal_dur
        c = 0.5 * (1.0 + math.cos(math.pi * progress)) # smooth cosine from 1 to 0
        return a + (b - a) * (1 - c)

    return thunk
