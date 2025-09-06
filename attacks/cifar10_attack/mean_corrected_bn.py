import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Corrected_mu_BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_mean, running_var, weight, bias, training, momentum, eps):
        weight = weight[None, :, None, None]
        bias = bias[None, :, None, None]
        eps = torch.tensor([eps]).to(dtype=x.dtype, device=x.device)
        n, c, h, w = x.shape

        sigmab, mub = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=True)
        sigmamub, mumub = torch.var_mean(mub, unbiased=False)
        summub = (mub ** 2).sum()
        sigmamub_summub = (c - 2)* sigmamub * c/(c+1) / summub
        mujs = (1 - sigmamub_summub) * mub
        invstd = (sigmab + eps).rsqrt_()

        xhat = (x - mujs) * invstd
        y = xhat * weight + bias
        ctx.save_for_backward(x, mub, sigmab, mumub, sigmamub, mujs, invstd, xhat, weight, eps)
        return y
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dl_y):
        x, mub, sigmab, mumub, sigmamub, mujs, invstd, xhat, gamma, eps = ctx.saved_tensors
        n, c, h, w = x.shape
        N = n * h * w
        dl_beta = dl_y.sum(dim=[0, 2, 3])
        dl_gamma = (dl_y * xhat).sum(dim=[0, 2, 3])
        dl_xhat = dl_y * gamma

        dxhat_x = invstd
        dxhat_mujs = -invstd
        dxhat_sigmab = -0.5 * (x - mujs) * (sigmab + eps) ** -1.5

        dmujs_mub = 1 - (c - 2)*c/(c+1) * sigmamub / (mub ** 2).sum()
        dmujs_summub = mub * (c - 2)*c/(c+1) * sigmamub / (mub ** 2).sum() ** 2
        dsummub_mub = 2 * mub
        dmujs_sigmamub = - mub * (c - 2)*c/(c+1) / (mub ** 2).sum()
        dmub_x = 1 / N
        dsigmab_x = 2 * (x - mub) / N
        dsigmamub_mub = 2 * (mub - mumub) / c

        dl_sigmab = (dl_xhat * dxhat_sigmab).sum(dim=[0, 2, 3]).reshape(1, c, 1, 1)
        dl_mujs = (dl_xhat * dxhat_mujs).sum(dim=[0, 2, 3]).reshape(1, c, 1, 1)
        dl_sigmamub = (dl_mujs * dmujs_sigmamub).sum()
        dl_summub = (dl_mujs * dmujs_summub).sum()
        dl_mub = dl_mujs * dmujs_mub + dl_summub * dsummub_mub + dl_sigmamub * dsigmamub_mub 
        dl_x = dl_xhat * dxhat_x + dl_mub * dmub_x + dl_sigmab * dsigmab_x

        return dl_x, dl_gamma, dl_beta, None, None, None, None, None
    

class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, x):
        return Corrected_mu_BatchNorm.apply(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)


def convert_batchnorm_mean(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm_mean(child))
    del module
    return module_output