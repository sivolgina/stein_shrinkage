import os
from evaluate import *

def init_log_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, 'w') as f:
            f.write('Epoch\tLoss_Train\tP_Clean\tS_Clean\tF1_Clean\tJS_Clean\n')

def log_epoch(path, epoch, results):
    with open(path, 'a') as f:
        f.write(f'{epoch}\t' + '\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in results) + '\n')

def log_test(path, epsilon, metrics):
    with open(path, 'a') as f:
        f.write(
            f'FINAL_TEST_{epsilon:.2f}\t-\t-\t-\t-\t-\t-\t'
            f'{metrics["Precision"]:.4f}\t'
            f'{metrics["Sensitivity"]:.4f}\t'
            f'{metrics["F1"]:.4f}\t'
            f'{metrics["js"]:.4f}\n'
        )



