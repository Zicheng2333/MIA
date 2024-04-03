import abc
from collections import namedtuple

import torch
import torch.nn as nn

import typing
from . import function
from ..dependency import Group

import tqdm
from copy import deepcopy

__all__ = [
    # Base Class
    "Importance",

    # Basic Group Importance
    "GroupNormImportance",
    "GroupTaylorImportance",
    "GroupHessianImportance",

    # Aliases
    "MagnitudeImportance",
    "TaylorImportance",
    "HessianImportance",

    # Other Importance
    "BNScaleImportance",
    "LAMPImportance",
    "RandomImportance",

    'DeltaLossImportance',

]

class Importance(abc.ABC): #abc.ABC：抽象基类
    """ Estimate the importance of a tp.Dependency.Group, and return an 1-D per-channel importance score.

        It should accept a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        Just ignore the ch_groups if you are not familar with grouping.

        Example:
            ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
            scorer = MagnitudeImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
            ``` 
    """
    @abc.abstractclassmethod
    def __call__(self, group: Group) -> torch.Tensor: 
        raise NotImplementedError


class GroupNormImportance(Importance):
    """ A general implementation of magnitude importance. By default, it calculates the group L2-norm for each channel/dim.
        It supports several variants like:
            - Standard L1-norm of the first layer in a group: MagnitudeImportance(p=1, normalizer=None, group_reduction="first")
            - Group L1-Norm: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean")
            - BN Scaling Factor: MagnitudeImportance(p=1, normalizer=None, group_reduction="mean", target_types=[nn.modules.batchnorm._BatchNorm])

        Args:
            * p (int): the norm degree. Default: 2
            * group_reduction (str): the reduction method for group importance. Default: "mean"
            * normalizer (str): the normalization method for group importance. Default: "mean"
            * target_types (list): the target types for importance calculation. Default: [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]

        Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupNormImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 p: int=2, 
                 group_reduction: str="mean", 
                 normalizer: str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

    def _lamp(self, imp): # Layer-adaptive Sparsity for the Magnitude-based Pruning
        argsort_idx = torch.argsort(imp, dim=0, descending=True)
        sorted_imp = imp[argsort_idx.tolist()]
        cumsum_imp = torch.cumsum(sorted_imp, dim=0)
        sorted_imp = sorted_imp / cumsum_imp
        inversed_idx = torch.argsort(argsort_idx).tolist()  # [0, 1, 2, 3, ..., ]
        return sorted_imp[inversed_idx]
    
    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        elif normalizer.startswith('sentinel'): # normalize the score with the k-th smallest element. e.g. sentinel_0.5 means median normalization
            sentinel = float(normalizer.split('_')[1]) * len(group_importance)
            sentinel = torch.argsort(group_importance, dim=0, descending=False)[int(sentinel)]
            return group_importance / (group_importance[sentinel]+1e-8)
        elif normalizer=='lamp':
            return self._lamp(group_importance)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0])
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0]) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp) # accumulated importance
            elif self.group_reduction == "max": # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod": # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp)-1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0) # no reduction
            else:
                raise NotImplementedError
        
        if self.group_reduction == "mean":
            reduced_imp /= len(group_imp)
        return reduced_imp
    
    @torch.no_grad()
    def __call__(self, group: Group):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    local_imp = layer.bias.data[idxs].abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                local_imp = w.abs().pow(self.p).sum(1)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                
                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            ####################
            # BatchNorm
            ####################
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            ####################
            # LayerNorm
            ####################
            elif prune_fn == function.prune_layernorm_out_channels:

                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    local_imp = w.abs().pow(self.p)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        local_imp = layer.bias.data[idxs].abs().pow(self.p)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)

        return group_imp


class BNScaleImportance(GroupNormImportance):
    """Learning Efficient Convolutional Networks through Network Slimming, 
    https://arxiv.org/abs/1708.06519

    Example:
    
        It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
        All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
        
        ```python
            DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
            group = DG.get_pruning_group( model.bn1, tp.prune_batchnorm_out_channels, idxs=[2, 6, 9] )    
            scorer = BNScaleImportance()    
            imp_score = scorer(group)    
            #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
            min_score = imp_score.min() 
        ``` 

    """

    def __init__(self, group_reduction='mean', normalizer='mean'):
        super().__init__(p=1, group_reduction=group_reduction, normalizer=normalizer, bias=False, target_types=(nn.modules.batchnorm._BatchNorm,))


class LAMPImportance(GroupNormImportance):
    """Layer-adaptive Sparsity for the Magnitude-based Pruning,
    https://arxiv.org/abs/2010.07611

    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = LAMPImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """

    def __init__(self, p=2, group_reduction="mean", normalizer='lamp', bias=False):
        assert normalizer == 'lamp'
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

class RandomImportance(Importance):
    """ Random importance estimator
    Example:
    
            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = RandomImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    @torch.no_grad()
    def __call__(self, group, **kwargs):
        _, idxs = group[0]
        re = torch.rand(len(idxs))

        return re


class GroupTaylorImportance(GroupNormImportance):
    """ Grouped first-order taylor expansion of the loss function.
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf

        Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                loss = loss_fn(model(inputs), labels)
                loss.backward() # compute gradients
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                scorer = GroupTaylorImportance()    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 multivariable:bool=False, 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue
            
            # Conv/Linear Output
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv/Linear Input
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                    dw = (layer.weight.grad).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                
                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)
                local_imp = local_imp[idxs]

                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w*dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

class GroupHessianImportance(GroupNormImportance):
    """Grouped Optimal Brain Damage:
       https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

       Example:

            It accepts a group as inputs, and return a 1-D tensor with the same length as the number of channels.
            All groups must be pruned simultaneously and thus their importance should be accumulated across channel groups.
            
            ```python
                inputs, labels = ...
                DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224)) 
                scorer = GroupHessianImportance()   
                scorer.zero_grad() # clean the acuumulated gradients if necessary
                loss = loss_fn(model(inputs), labels, reduction='none') # compute loss for each sample
                for l in loss:
                    model.zero_grad() # clean the model gradients
                    l.backward(retain_graph=True) # compute gradients for each sample
                    scorer.accumulate_grad(model) # accumulate gradients of each sample
                group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )    
                imp_score = scorer(group)    
                #imp_score is a 1-D tensor with length 3 for channels [2, 6, 9]  
                min_score = imp_score.min() 
            ``` 
    """
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._accu_grad = {}
        self._counter = {}

    def zero_grad(self):
        self._accu_grad = {}
        self._counter = {}

    def accumulate_grad(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad:
                    self._accu_grad[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter:
                    self._counter[param] = 1
                else:
                    self._counter[param] += 1
    
    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad.items():
                p.grad.data = g / self._counter[p]
            self.zero_grad()

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)
                        h = (layer.weight.grad).flatten(1)
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                        local_imp = local_imp.repeat(layer.groups)
                    local_imp = local_imp[idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            
            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine:
                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
                    if self.bias and layer.bias is not None and layer.bias.grad is not None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)
            

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp


# Aliases
class MagnitudeImportance(GroupNormImportance):
    pass

class TaylorImportance(GroupTaylorImportance):
    pass

class HessianImportance(GroupHessianImportance):
    pass


_helpers = namedtuple('GroupItem', ['dep', 'idxs'])

class DeltaLossImportance(Importance):
    def __init__(self,model,val_loader,device,
                 target_types: list = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.LayerNorm]):
        self.model = model
        self.val_loader = val_loader

        self.device = device
        self.target_types = target_types

    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (
                        group_importance.max() - group_importance.min() + 1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std() + 1e-8)
        elif normalizer.startswith(
                'sentinel'):  # normalize the score with the k-th smallest element. e.g. sentinel_0.5 means median normalization
            sentinel = float(normalizer.split('_')[1]) * len(group_importance)
            sentinel = torch.argsort(group_importance, dim=0, descending=False)[int(sentinel)]
            return group_importance / (group_importance[sentinel] + 1e-8)

        else:
            raise NotImplementedError

    def _reduce(self, group_imp: typing.List[torch.Tensor], group_idxs: typing.List[typing.List[int]]):
        if len(group_imp) == 0: return group_imp
        if self.group_reduction == 'prod':
            reduced_imp = torch.ones_like(group_imp[0])
        elif self.group_reduction == 'max':
            reduced_imp = torch.ones_like(group_imp[0]) * -99999
        else:
            reduced_imp = torch.zeros_like(group_imp[0])

        for i, (imp, root_idxs) in enumerate(zip(group_imp, group_idxs)):
            if self.group_reduction == "sum" or self.group_reduction == "mean":
                reduced_imp.scatter_add_(0, torch.tensor(root_idxs, device=imp.device), imp)  # accumulated importance
            elif self.group_reduction == "max":  # keep the max importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                selected_imp = torch.maximum(input=selected_imp, other=imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == "prod":  # product of importance
                selected_imp = torch.index_select(reduced_imp, 0, torch.tensor(root_idxs, device=imp.device))
                torch.mul(selected_imp, imp, out=selected_imp)
                reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), selected_imp)
            elif self.group_reduction == 'first':
                if i == 0:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction == 'gate':
                if i == len(group_imp) - 1:
                    reduced_imp.scatter_(0, torch.tensor(root_idxs, device=imp.device), imp)
            elif self.group_reduction is None:
                reduced_imp = torch.stack(group_imp, dim=0)  # no reduction
            else:
                raise NotImplementedError

        if self.group_reduction == "mean":
            reduced_imp /= len(group_imp)
        return reduced_imp

    def evaluate_loss(self, model):
        model.eval()  # 设置模型为评估模式
        total_loss = 0.0
        with torch.no_grad():  # 不计算梯度
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    @torch.no_grad()
    def __call__(self, group: Group):
        # 计算原始模型在验证集上的损失
        original_loss = self.evaluate_loss(self.model)

        group_imp = []
        group_idxs = []
        # 遍历group中的每个依赖项和索引
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue

            print('evaluating layer:',layer)
            print('idxs:',idxs)

            for idx in idxs:
                original_param = layer.weight.data[idx].clone()
                layer.weight.data[idx] = 0

                pruned_loss = self.evaluate_loss(self.model)

                # 计算损失变化作为重要性分数
                loss_change = original_loss-pruned_loss
                group_imp.append(torch.tensor([loss_change]))
                group_idxs.append(root_idxs)

                # 恢复原始参数
                #print('idx:',idx)
                #print('weight:',layer.weight.data)
                layer.weight.data[idx] = original_param


        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp,group_idxs)
        group_imp = self._normalize(group_imp,'mean')

        return group_imp



