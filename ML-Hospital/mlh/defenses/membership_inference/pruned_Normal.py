import torch

import os

import torch.nn.functional as F
from mlh.defenses.membership_inference.trainer import Trainer
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.utils.prune

import mlh.defenses.torch_pruning as tp
from functools import partial


from mlh import utils

class TrainTargetNormal(Trainer):

    def __init__(self, args, model, device, num_class, epochs=100,  log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs

        self.model = self.model.to(self.device)

        self.log_path = log_path

        self.args = args

    def progressive_pruning(self, pruner, model, speed_up, example_inputs):
        model.eval()
        base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = 1
        while current_speed_up < speed_up:
            pruner.step(interactive=False)
            pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            current_speed_up = float(base_ops) / pruned_ops
            if pruner.current_step == pruner.iterative_steps:
                break
            # print(current_speed_up)
        return current_speed_up

    def get_pruner(self, model, example_inputs):
        self.args.sparsity_learning = False
        if self.args.method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.args.global_pruning)
        elif self.args.method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=self.args.global_pruning)
        elif self.args.method == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=self.args.global_pruning)
        elif self.args.method == "slim":
            self.args.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.args.reg, global_pruning=self.args.global_pruning)
        elif self.args.method == "group_slim":
            self.args.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=self.args.reg, global_pruning=self.args.global_pruning,
                                   group_lasso=True)
        elif self.args.method == "group_norm":
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=self.args.global_pruning)
        elif self.args.method == "group_sl":
            self.args.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2, normalizer='max')
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=self.args.reg, global_pruning=self.args.global_pruning)
        elif self.args.method == "growing_reg":
            self.args.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=self.args.reg, delta_reg=self.args.delta_reg,
                                   global_pruning=self.args.global_pruning)
        else:
            raise NotImplementedError

        # args.is_accum_importance = is_accum_importance
        unwrapped_parameters = []
        ignored_layers = []
        pruning_ratio_dict = {}
        # ignore output layers
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.num_class:
                ignored_layers.append(m)
            elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == self.num_class:
                ignored_layers.append(m)

        # Here we fix iterative_steps=200 to prune the model progressively with small steps
        # until the required speed up is achieved.
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=self.args.iterative_steps,
            pruning_ratio=1.0,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=self.args.max_pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
        return pruner

    def eval(self,  test_loader):
        correct = 0
        total = 0
        loss = 0

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss += nn.functional.cross_entropy(out,target,reduction='sum')
                pred = out.max(1)[1]
                correct += (pred == target).sum()
                total += len(target)
        return (correct / total).item(), (loss / total).item()

    def train(self,
            model,
            train_loader,
            test_loader,
            epochs,
            lr,
            lr_decay_milestones,
            lr_decay_gamma=0.1,
            save_as=None,

            # For pruning
            weight_decay=5e-4,
            save_state_dict_only=True,
            pruner=None,

    ):
        device = self.device
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay if pruner is None else 0,
        )
        milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_gamma
        )
        model.to(device)
        best_acc = -1

        for epoch in range(epochs):
            model.train()

            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.cross_entropy(out, target)
                loss.backward()
                if pruner is not None:
                    pruner.regularize(model)  # for sparsity learning
                optimizer.step()

            if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
                pruner.update_reg()  # increase the strength of regularization
                # print(pruner.group_reg[pruner._groups[0]])

            model.eval()
            acc, val_loss = self.eval(test_loader)
            self.args.logger.info(
                "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                    epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
                )
            )
            if best_acc < acc:
                os.makedirs(self.log_path, exist_ok=True)
                if self.args.mode == "prune":
                    if save_as is None:
                        save_as = os.path.join(self.log_path,
                                               "{}_{}_{}.pth".format(self.args.dataset, self.args.model,
                                                                     self.args.method))

                    if save_state_dict_only:
                        torch.save(model.state_dict(), save_as)
                    else:
                        torch.save(model, save_as)
                elif self.args.mode == "pretrain":
                    if save_as is None:
                        save_as = os.path.join(self.log_path,
                                               "{}_{}.pth".format(self.args.dataset, self.args.model))
                    torch.save(model.state_dict(), save_as)
                best_acc = acc
            scheduler.step()

        self.args.logger.info("Best Acc=%.4f" % (best_acc))

    def train_pruned_model(self, train_loader, test_loader):
        if self.args.mode == "prune":
            prefix = 'global' if self.args.global_pruning else 'local'  # 全局或局部剪枝
            logger_name = "{}-{}-{}-{}".format(self.args.dataset, prefix, self.args.method, self.args.model)
            self.log_path = os.path.join(self.log_path, self.args.dataset, self.args.mode, logger_name)
            log_file = "{}/{}.txt".format(self.log_path, logger_name)
        elif self.args.mode == "pretrain":
            self.log_path = os.path.join(self.log_path, self.args.dataset, self.args.mode)
            logger_name = "{}-{}".format(self.args.dataset, self.args.model)
            log_file = "{}/{}.txt".format(self.log_path, logger_name)


        images, _ = next(iter(train_loader))
        example_input = images[0].unsqueeze(0).to(self.device)

        if self.args.mode == 'prune':
            pruner = self.get_pruner(self.model, example_inputs=example_input)

            # TODO 0. Sparsity Learning

            if self.args.sparsity_learning:
                print("###################Start sparsity learning###################")
                reg_pth = "reg_{}_{}_{}_{}.pth".format(self.args.dataset, self.args.model, self.args.method,
                                                       self.args.reg)
                reg_pth = os.path.join(os.path.join(self.log_path, reg_pth))
                if not self.args.sl_restore:
                    self.args.logger.info("Regularizing...")

                    self.train(
                        self.model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        epochs=self.args.sl_total_epochs,
                        lr=self.args.sl_lr,
                        lr_decay_milestones=self.args.sl_lr_decay_milestones,
                        lr_decay_gamma=self.args.lr_decay_gamma,
                        pruner=pruner,
                        save_state_dict_only=True,
                        save_as=reg_pth,
                    )
                self.args.logger.info("Loading the sparse model from {}...".format(reg_pth))
                self.model.load_state_dict(torch.load(reg_pth, map_location=self.device))

            print("###################Start pruning###################")
            # TODO 1. Pruning
            self.model.eval()
            ori_ops, ori_size = tp.utils.count_ops_and_params(self.model, example_inputs=example_input)
            ori_acc, ori_val_loss = self.eval(test_loader)
            self.args.logger.info("Pruning...")
            self.progressive_pruning(pruner, self.model, self.args.speed_up, example_input)
            del pruner  # remove reference
            self.args.logger.info(self.model)
            pruned_ops, pruned_size = tp.utils.count_ops_and_params(self.model, example_inputs=example_input)
            pruned_acc, pruned_val_loss = self.eval(test_loader)

            self.args.logger.info(
                "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                    ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
                )
            )
            self.args.logger.info(
                "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                    ori_ops / 1e6,
                    pruned_ops / 1e6,
                    pruned_ops / ori_ops * 100,
                    ori_ops / pruned_ops,
                )
            )
            self.args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
            self.args.logger.info(
                "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
            )

            print("###################Start fine-tuning###################")
            # TODO 2. Finetuning
            self.args.logger.info("Finetuning...")
            self.train(
                self.model,
                epochs=self.epochs,
                lr=self.args.lr,
                lr_decay_milestones=self.args.lr_decay_milestones,
                train_loader=train_loader,
                test_loader=test_loader,
                save_state_dict_only=False,
            )

        elif self.args.mode == 'pretrain':
            print("###################Start pretraining###################")
            ops, params = tp.utils.count_ops_and_params(
                self.model, example_inputs=example_input,
            )
            self.args.logger.info("Params: {:.2f} M".format(params / 1e6))
            self.args.logger.info("ops: {:.2f} M".format(ops / 1e6))
            self.train(
                model=self.model,
                epochs=self.epochs,
                lr=self.args.lr,
                lr_decay_milestones=self.args.lr_decay_milestones,
                train_loader=train_loader,
                test_loader=test_loader
            )
        elif self.args.mode == 'eval':
            self.model.eval()
            ops, params = tp.utils.count_ops_and_params(
                self.model, example_inputs=example_input,
            )
            print("Params: {:.2f} M".format(params / 1e6))
            print("ops: {:.2f} M".format(ops / 1e6))
            val_acc, val_loss = eval(self.model, test_loader)
            train_acc, train_loss = eval(self.model,train_loader)

            print("Train Acc: {:.4f} Train Loss: {:.4f} Val Acc: {:.4f} Val Loss: {:.4f}\n".format(train_acc, train_loss, val_acc, val_loss))





