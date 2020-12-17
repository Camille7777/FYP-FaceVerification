import torch


def fusion_simple_append(output, soft_bio):
    result = torch.cat([output, soft_bio], dim=1)
    return result


def fusion_kronecker_product(output, soft_bio):
    result = torch.einsum("ab,ac->abc", output, soft_bio)
    result = result.view(output.size(0), output.size(1) * soft_bio.size(1))
    return result


def fusion_none(*args):
    return args[0]


fusion_dict = {"append": fusion_simple_append,
               "kronecker": fusion_kronecker_product}


def get_fusion_result(method, output, soft_bio):
    return fusion_dict.get(method, fusion_none)(output, soft_bio)

