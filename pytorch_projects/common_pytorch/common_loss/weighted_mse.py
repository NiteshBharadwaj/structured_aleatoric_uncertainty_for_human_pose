import torch
import numpy as np

def weighted_mse_loss(input, target, weights, size_average):
    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_l1_loss(input, target, weights, size_average):
    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()


def weighted_mse_loss_u(input, target, weights, size_average, uncer, uncer2):
    uncer_concat = torch.cat((uncer, uncer2), 1)
    q_list = [make_upper_triangular(m)
              for m in torch.unbind(uncer_concat, dim=0)]
    batch_q = torch.stack(q_list, dim=0)
    out = (target - input)
    out = out * weights
    out = torch.reshape(out, (-1, 1, 48))
    res = torch.bmm(out, batch_q)
    res = torch.bmm(res, torch.transpose(res, 1, 2))

    # Calculating determinant
    det = torch.sum(uncer * weights, 1) * 2.0
    loss = (res.sum() + det.sum()) / len(input)

    return loss

def make_upper_triangular(tensor_array,cuda = True):
    mat = torch.zeros(48, 48)
    if cuda:
        mat = mat.cuda()
    #mat[(dim1, dim2)] = tensor_array[48:]
    mat[np.tril_indices(48,-1)] = tensor_array[48:]
    mat[np.diag_indices(48)] = torch.exp(-1.0*tensor_array[:48])
    return mat