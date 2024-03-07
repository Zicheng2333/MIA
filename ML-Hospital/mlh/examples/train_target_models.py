import os
import sys
#sys.path.append("..")
#sys.path.append("../..")

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from mlh.defenses.membership_inference.AdvReg import TrainTargetAdvReg
from mlh.defenses.membership_inference.DPSGD import TrainTargetDP
from mlh.defenses.membership_inference.LabelSmoothing import TrainTargetLabelSmoothing
from mlh.defenses.membership_inference.MixupMMD import TrainTargetMixupMMD
from mlh.defenses.membership_inference.PATE import TrainTargetPATE

#from mlh.defenses.membership_inference.Normal import TrainTargetNormal
from mlh.defenses.membership_inference.pruned_Normal import TrainTargetNormal

import torch

from mlh.data_preprocessing.data_loader_vit import GetDataLoader

import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)

from models import vit
from models import resnet

from mlh import utils



def get_target_model(name="vit_b_16", num_classes=1000,resume=False):
    if name == "vit_b_16":
        model = vit.VisionTransformer(num_classes=num_classes)
    elif name == 'resnet18':
        model = resnet.ResNet18(num_classes=num_classes)
    elif name == 'resnet34':
        model = resnet.ResNet34(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet.ResNet50(num_classes=num_classes)
    elif name == 'resnet152':
        model =resnet.ResNet152(num_classes=num_classes)

    else:
        raise ValueError("model not supported")


    if resume:
            print("resume!")
    
    return model


if __name__ == "__main__":

    opt = utils.parse_args()
    s = GetDataLoader(opt)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()

    if opt.mode == "target":
        train_loader, inference_loader, test_loader = target_train_loader, target_inference_loader, target_test_loader,
    elif opt.mode == "shadow":
        train_loader, inference_loader, test_loader = shadow_train_loader, shadow_inference_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")

    target_model = get_target_model(name=opt.model, num_classes=opt.num_class,resume=opt.resume)


    save_pth = f'{opt.log_path}/{opt.dataset}/{opt.training_type}/{opt.mode}'

    if opt.training_type == "Normal_f_vit_bt":

        total_evaluator = TrainTargetNormal(
            model=target_model,model_name=opt.model,device=opt.device,num_class=opt.num_class, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "LabelSmoothing":

        total_evaluator = TrainTargetLabelSmoothing(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "AdvReg":

        total_evaluator = TrainTargetAdvReg(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        model = total_evaluator.model

    elif opt.training_type == "DP":
        total_evaluator = TrainTargetDP(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "MixupMMD":

        target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted = s.get_sorted_data_mixup_mmd()
        if opt.mode == "target":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = target_train_sorted_loader, target_inference_sorted_loader, start_index_target_inference, target_inference_sorted

        elif opt.mode == "shadow":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_shadow_inference, shadow_inference_sorted

        total_evaluator = TrainTargetMixupMMD(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, train_loader_ordered,
                              inference_loader_ordered, test_loader, starting_index, inference_sorted)

    elif opt.training_type == "PATE":

        total_evaluator = TrainTargetPATE(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)

    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE")

    print("Finish Training")
