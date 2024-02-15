

import abc
from models.attack_model import MLP_BLACKBOX
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from runx.logx import logx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class Trainer(abc.ABC):
    """
    Abstract base class for MIA defenses.
    """

    def __init__(self,):

        super().__init__()

    @abc.abstractmethod
    def train(self, dataloader, train_epoch, **kwargs):
        """
        Infer membership status of samples from the target estimator. This method
        should be overridden by all concrete inference attack implementations.

        :param dataloader: An array with reference inputs to be used in the attack.
        :param y: Labels for `x`. This parameter is only used by some of the attacks.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status (1 indicates member of training set,
                 0 indicates non-member) or class probabilities.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def infer(self, datalaoder, **kwargs):
    #     raise NotImplementedError
