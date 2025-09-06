import os
from evaluate import *

def init_log_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, 'w') as f:
            f.write('Epoch\tLoss_Train\tP_Clean\tS_Clean\tF1_Clean\tJS_Clean\tP_0.03\tS_0.03\tF1_0.03\tJS_0.03\tP_0.1\tS_0.1\tF1_0.1\tJS_0.1\tP_0.15\tS_0.15\tF1_0.15\tJS_0.15\tP_0.2\tS_0.2\tF1_0.2\tJS_0.2\tP_0.3\tS_0.3\tF1_0.3\tJS_0.3\n')

def log_epoch(path, epoch, results):
    with open(path, 'a') as f:
        f.write(f'{epoch}\t' + '\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in results) + '\n')

def log_test(path, model, test_noise_loaders, epsilons):
    with open(path, 'a') as f:
        for loader, eps in zip(test_noise_loaders, epsilons):
            metrics = validate(loader, model)
            f.write(
                f'FINAL_TEST_{eps}\t-\t-\t-\t-\t-\t-\t'
                f'{metrics["Precision"]:.4f}\t{metrics["Senstivity"]:.4f}\t'
                f'{metrics["F1"]:.4f}\t{metrics["js"]:.4f}\n'
            )
        f.flush()



