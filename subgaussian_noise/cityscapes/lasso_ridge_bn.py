import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, lambda_=0.3, mode='lasso'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.lambda_ = lambda_
        self.mode = mode.lower() 

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            n, c, h, w = x.shape
            N = n * h * w
            if self.mode == 'ridge':
                mean = mean * N/(N + self.lambda_)
                var = var / (1 + self.lambda_)
            elif self.mode == 'lasso':
                mean = torch.sign(mean) * torch.clamp(torch.abs(mean) - self.lambda_ / (2* N), min=0.0)
                var = torch.clamp(var - self.lambda_ / 2, min=0.0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x_hat

def convert_batchnorm_lasso(model, lambda_=1e-3, mode='lasso'):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_bn = RegBatchNorm2d(
                num_features=module.num_features,
                lambda_=lambda_,
                mode=mode,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            setattr(model, name, new_bn)
        else:
            convert_batchnorm_lasso(module, lambda_, mode)
    return model 

def convert_batchnorm_ridge(model, lambda_=1e-3, mode='ridge'):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_bn = RegBatchNorm2d(
                num_features=module.num_features,
                lambda_=lambda_,
                mode=mode,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            setattr(model, name, new_bn)
        else:
            convert_batchnorm_ridge(module, lambda_, mode)
    return model 
