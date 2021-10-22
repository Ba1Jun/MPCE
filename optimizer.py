from torch.optim import Adam, SGD
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.nn.utils import clip_grad_norm_
from log_utils import logger

class Optim(object):
    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, max_weight_value=None, lr_decay=1,
                 lr_decay_patience=6):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.metric_history = []
        self.patience = 0

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)
    @property
    def best_metric(self):
        if len(self.metric_history) == 0:
            return None
        return min(self.metric_history)

    def is_better(self, metric):
        if len(self.metric_history) == 0:
            return True
        if metric < min(self.metric_history):
            return True
        return False

    def update_lr(self, metric):
        # self.last_ppl = ppl
        hit_trial = False
        if self.is_better(metric):
            self.patience = 0
        else:
            self.patience += 1
            if self.patience >= self.lr_decay_patience:
                if self.lr >= 1e-6:
                    self.lr = self.lr * self.lr_decay
                self.patience = 0
                hit_trial = True


        self.optimizer.param_groups[0]['lr'] = self.lr
        return  hit_trial
