import torch


def fusion_simple_append(output, soft_bio):
    result = torch.cat([output, soft_bio], dim=1)
    return result


def fusion_none(*args):
    return args[0]


fusion_dict = {"append": fusion_simple_append}


def get_fusion_result(method, output, soft_bio):
    return fusion_dict.get(method, fusion_none)(output, soft_bio)

