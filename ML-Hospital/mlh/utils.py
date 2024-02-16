

import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')


    parser.add_argument('--mode', type=str, default="shadow",
                        help='target, shadow')
    parser.add_argument('--resume', type=bool, default=False,
                        help='True,False ')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='vit_b_16')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal_f_vit_bt",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    #parser.add_argument('--inference-dataset', type=str, default='ImageNet1K',
                       # help='if yes, load pretrained attack model to inference')
    parser.add_argument('--inference-dataset', type=str, default='ImageNet',
                        help='if yes, load pretrained attack model to inference')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data-path', type=str, default='/data/dataset/imagenet/images',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="256,256,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='')

    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
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
