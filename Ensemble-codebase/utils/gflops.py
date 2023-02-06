# Note: the gflops.py is experimental and may be inaccurate. If you experience any problems, please tell us.
# Please run this in project directory: python -m utils.gflops

import torch
from torchvision.models import resnet50
from thop import profile
import sys
import model.model as models
import argparse
parser = argparse.ArgumentParser()

# Examples:
# ImageNet-LT
# ResNeXt50
# python -m utils.gflops ResNeXt50Model 0 --num_experts 4 --reduce_dim True --use_norm False

# iNaturalist
## LDAM
# python -m utils.gflops ResNet50Model 1 --num_experts 3 --reduce_dim True --use_norm True

# Imbalance CIFAR 100
## LDAM
# python -m utils.gflops ResNet32Model 2 --num_experts 3 --reduce_dim True --use_norm True


parser.add_argument("model_name", type=str)
parser.add_argument("dataset", type=str, help="0: ImageNet-LT, 1: iNaturalist, 2: Imbalance CIFAR 100")
parser.add_argument("--num_experts", type=int, default=1)
parser.add_argument("--layer2_dim", type=int, default=0)
parser.add_argument("--layer3_dim", type=int, default=0)
parser.add_argument("--layer4_dim", type=int, default=0)
parser.add_argument("--reduce_dim", type=str, default="False", help="True: reduce dimension")
parser.add_argument("--use_norm", type=str, default="False", help="True: use_norm")
parser.add_argument("--ea_percentage", type=str, default=None, help="Percentage of passing each expert: only use this if you are calculating GFLOPs for an EA module. Example: 40.99,9.47,49.54")

args = parser.parse_args()

model_name = args.model_name
num_classes_dict = {
    "0": 1000,
    "1": 8142,
    "2": 100
}
dataset_name_dict = {
    "0": "ImageNet-LT",
    "1": "iNaturalist 2018",
    "2": "Imbalanced CIFAR 100"
}
print("Using dataset", dataset_name_dict[args.dataset])

num_classes = num_classes_dict[args.dataset]

def gflops_normed_linear(m, x, y):
    # per output element
    num_instance = y.size(0)
    total_ops = m.weight.numel() * num_instance + m.weight.size(0) # weight normalization can be ignored
    m.total_ops += torch.DoubleTensor([int(total_ops)])

num_experts = args.num_experts
layer2_dim = args.layer2_dim
layer3_dim = args.layer3_dim
layer4_dim = args.layer4_dim
reduced_dim = True if args.reduce_dim == "True" else False
use_norm = True if args.use_norm == "True" else False
ea_percentage = args.ea_percentage
if ea_percentage is not None:
    ea_percentage = [float(item) for item in ea_percentage.split(",")]
    force_all = True
    loop = range(num_experts)
else:
    force_all = False
    loop = [num_experts-1]

total_macs = 0

for i in loop: # i: num_experts - 1 so we need to add one
    model_arg = {
        "num_classes": num_classes,
        "num_experts": i+1,
        **({"layer2_output_dim": layer2_dim} if layer2_dim else {}),
        **({"layer3_output_dim": layer3_dim} if layer3_dim else {}),
        **({"layer4_output_dim": layer4_dim} if layer4_dim else {}),
        **({"reduce_dimension": reduced_dim} if reduced_dim else {}),
        **({"use_norm": use_norm} if use_norm else {}),
        **({"force_all": force_all} if force_all else {})
    }
    
    print("Model Name: {}, Model Arg: {}".format(model_name, model_arg))

    model = (getattr(models, model_name))(**model_arg)

    model = model.backbone
    model = model.eval()
    model = model

    if num_classes == 10 or num_classes == 100: # model inputs are different for CIFAR
        input = torch.randn(1, 3, 32, 32)
    else:
        input = torch.randn(1, 3, 224, 224)

    input_dim = input.shape
    print("Using input size", input_dim)
    macs, _ = profile(model, inputs=(input, ), verbose=False, custom_ops={
        models.resnet_cifar.NormedLinear: gflops_normed_linear,
        models.ea_resnet_cifar.NormedLinear: gflops_normed_linear,
        models.ResNet.NormedLinear: gflops_normed_linear,
        models.EAResNet.NormedLinear: gflops_normed_linear
    })
    if force_all:
        percentage_curr = ea_percentage[i]
        total_macs += percentage_curr * macs / 100
    else:
        total_macs += macs

print("macs(G):", total_macs/1000/1000/1000)
print()
