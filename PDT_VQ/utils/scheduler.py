import contextlib

import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn

class Scheduler:
    """updates the learning rate, decides whether to stop and stores the model
    if it is the best one so far"""

    def __init__(self, lr0):
        self.lr = lr0  # will be updated depending on loss values
        self.verbose = True
        self.loss_values = []
        self.last_lr_update = 0

    def quiet(self):
        "not verbose and do not save models"
        self.verbose = False

    def append_loss(self, loss):
        self.loss_values.append(loss)
        loss_values = np.array(self.loss_values, dtype=float)
        epoch = len(loss_values)

        best_loss_value = loss_values.min()

        # check if we need to stop optimization
        if epoch > self.last_lr_update + 50 and np.all(
            loss_values[-50:] > best_loss_value
        ):
            if self.verbose:
                print("Val loss did not improve for 50 epochs, stopping")
            self.last_lr_update = epoch
            self.lr = 0
        elif epoch > self.last_lr_update + 10 and np.all(
            loss_values[-10:] > best_loss_value
        ):
            # check if we need to reduce the learning rate
            if self.verbose:
                print("Val loss did not improve for 10 epochs, reduce LR")
            self.last_lr_update = epoch
            self.lr /= 10
            if self.lr < 2e-6:
                if self.verbose:
                    print("LR too small, stopping")
                self.lr = 0

    def should_stop(self):
        return self.lr == 0