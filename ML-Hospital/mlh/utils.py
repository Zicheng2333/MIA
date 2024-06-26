

import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num of workers to use')


    parser.add_argument('--mia_mode', type=str, default="shadow",
                        help='target, shadow')
    parser.add_argument('--resume', type=bool, default=False,
                        help='True,False ')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='vit_b_16')
    parser.add_argument('--load_pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        help='dataset')
    parser.add_argument('--num_class', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal_f_vit_bt",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    #parser.add_argument('--inference-dataset', type=str, default='ImageNet1K',
                       # help='if yes, load pretrained attack model to inference')
    parser.add_argument('--inference_dataset', type=str, default='ImageNet',
                        help='if yes, load pretrained attack model to inference')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data_path', type=str, default='/mnt/sharedata/ssd/common/datasets',
                        help='data_path')
    parser.add_argument('--load_path', type=str, default='/data/home/xzc/ML-Hospital/MIA',
                        help='data_path')
    #parser.add_argument('--input-shape', type=str, default="256,256,3",
    #                    help='comma delimited input shape input')
    #TODO 匹配模型的输入尺寸

    parser.add_argument('--log_path', type=str,
                        default='./save', help='')


    #TODO Basic options
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "eval"])

    parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
    parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")


    #TODO For pruning
    parser.add_argument("--method", type=str, default=None) #pruning method
    parser.add_argument("--speed-up", type=float, default=2)
    parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)  #不完全移除权重，只是将其值减小
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--global-pruning", action="store_true", default=False)
    parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
    parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
    parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
    parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
    parser.add_argument("--sl-restore", type=str, default=None)
    parser.add_argument("--iterative-steps", default=400, type=int)

    parser.add_argument("--pruning-ratio", type=float, default=1.0)

    args = parser.parse_args()

    #args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args



def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        # multi-class, one-hot encoded
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2
        ):  # multi-class, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2
        ):  # binary, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 1:  # index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels


from contextlib import contextmanager
import logging
import os, sys
from termcolor import colored
import copy
import numpy as np
import torch


class MagnitudeRecover():
    def __init__(self, model, reg=1e-3):
        self.rec = {}
        self.reg = reg
        self.cnt = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                norm = p.pow(2).mean()
                self.rec[name] = norm

    def regularize(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.rec:
                    target_norm = self.rec[name]
                    if p.data.pow(2).mean() > target_norm:
                        self.rec.pop(name)
                        continue
                    p.grad.data += -self.reg * p.data
                    if self.cnt % 1000 == 0:
                        print(name, p.pow(2).mean(), target_norm)
        self.cnt += 1


def flatten_dict(dic):
    flattned = dict()

    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten(k, v)
                else:
                    _flatten(prefix + '/%s' % k, v)
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[prefix + '/%s' % k] = v

    _flatten(None, dic)
    return flattned


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log

        return prefix + " " + log


def get_logger(name='train', output=None, color=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # STDOUT
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        formatter = plain_formatter
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # FILE
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            filename = output
        else:
            os.makedirs(output, exist_ok=True)
            filename = os.path.join(output, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger