from boxinst_plugin.ops import pairwise_nlog
import torch
import torch.nn.functional as F


def unfold_wo_center(x, kernel_size, dilation):
    """
    :param x: [N, C, H, W]
    :param kernel_size: k
    :param dilation:
    :return: [N, C, K^2-1, H, W]
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat(
        (unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), dim=2)

    return unfolded_x


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    # this equation is equal to log(p_i * p_j + (1 - p_i) * (1 - p_j))
    # max is used to prevent overflow
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    return -log_same_prob[:, 0]


def test_pairwise():
    inp = torch.randn([2, 1, 8, 8]).cuda() * 10
    ref = compute_pairwise_term(inp, 3, 2)
    res = pairwise_nlog(inp, 3, 2)
    assert torch.allclose(ref, res, rtol=0.001, atol=0.001)
    inp.requires_grad_(True)
    torch.autograd.gradcheck(pairwise_nlog,
                             (inp, 3, 2),
                             eps=1e-3, atol=1e-3, rtol=1e-3)
