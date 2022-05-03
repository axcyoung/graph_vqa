import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = lr

    def step_and_update_lr(self, epoch):
        "Step with the inner optimizer"
        self._update_learning_rate(epoch)
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self, epoch, t=8):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale() * (0.5 ** (epoch // t))
        # lr = self.init_lr * (0.1 ** (epoch // t))
        # if epoch < 11:
        #     lr = self.init_lr * self._get_lr_scale() * (0.1 ** (epoch // t))
        # else:
        #     epoch -= 5
        #     lr = self.init_lr * self._get_lr_scale() * (0.1 ** (epoch // 5))

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

