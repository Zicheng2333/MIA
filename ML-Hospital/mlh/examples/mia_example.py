
import sys


# from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, LabelOnlyMIA, MetricBasedMIA
sys.path.append("..")
sys.path.append("../..")

from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA

import torch

from mlh.data_preprocessing.data_loader_pruning import GetDataLoader
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)

from models import vit
from models import resnet,resnet_tiny

from mlh import utils


def get_target_model(name="vit_b_16", num_classes=1000,resume=False):
    if name == "vit_b_16":
        model = vit.VisionTransformer(num_classes=num_classes)
    elif name == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes)
    elif name == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes)
    elif name == 'resnet152':
        model =resnet.resnet152(num_classes=num_classes)
    elif name == 'resnet56':
        model = resnet_tiny.resnet56(num_classes=num_classes)
    elif name == 'resnet8':
        model = resnet_tiny.resnet8(num_classes=num_classes)
    else:
        raise ValueError("model not supported")

    return model


if __name__ == "__main__":

    args = utils.parse_args()
    s = GetDataLoader(args)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()

    target_path = f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.dataset}/{args.mode}/{args.dataset}_{args.model}.pth'
    shadow_path = f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.dataset}/{args.mode}/{args.dataset}_{args.model}.pth'

    print(args.device)

    if args.mode == 'prune':
        print('pruned MIA!')
        # 加载目标模型
        checkpoint1 = torch.load(
            f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.dataset}/{args.mode}/{args.dataset}-global-{args.method}-{args.model}/{args.dataset}_{args.model}_{args.method}.pth')
        target_model = checkpoint1.to(args.device)
        target_model = torch.nn.DataParallel(target_model)

        # 加载影子模型
        checkpoint2 = torch.load(
            f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.dataset}/{args.mode}/{args.dataset}-global-{args.method}-{args.model}/{args.dataset}_{args.model}_{args.method}.pth')
        shadow_model = checkpoint2.to(args.device)
        shadow_model = torch.nn.DataParallel(shadow_model)

    elif args.mode == 'pretrain':
        print('pretrained MIA!')
        target_model = get_target_model(name=args.model, num_classes=args.num_class)
        shadow_model = get_target_model(name=args.model, num_classes=args.num_class)


        checkpoint1 = torch.load(
            f'{args.log_path}/{args.dataset}/{args.training_type}/target/{args.dataset}/{args.mode}/{args.dataset}_{args.model}.pth')
        target_model.load_state_dict(checkpoint1)
        target_model = target_model.to(args.device)
        target_model = torch.nn.DataParallel(target_model)
        print("target model loaded from {restore}!".format(restore=target_path))

        checkpoint2 = torch.load(
            f'{args.log_path}/{args.dataset}/{args.training_type}/shadow/{args.dataset}/{args.mode}/{args.dataset}_{args.model}.pth')
        shadow_model.load_state_dict(checkpoint2)
        shadow_model = shadow_model.to(args.device)
        shadow_model = torch.nn.DataParallel(shadow_model)
        print("shadow model loaded from {restore}!".format(restore=shadow_path))

    # load target/shadow model to conduct the attacks


    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    # attack_type = "metric-based"

    if attack_type == "label-only":
        attack_model = LabelOnlyMIA(
            device=args.device,
            target_model=target_model.eval(),
            shadow_model=shadow_model.eval(),
            target_loader=(target_train_loader, target_test_loader),
            shadow_loader=(shadow_train_loader, shadow_test_loader),
            input_shape=(3, 32, 32),
            nb_classes=10)
        auc = attack_model.Infer()
        print(auc)

    else:

        attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
                                       target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)

        # train attack model

        if "black-box" in attack_type:
            attack_model = BlackBoxMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
        elif "metric-based" in attack_type:
            attack_model = MetricBasedMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                batch_size=128)
