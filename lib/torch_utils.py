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
