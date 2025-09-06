import os

def init_log_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, 'w') as f:
            f.write('Epoch\tTrain Loss\tTrain Accuracy\tVal Clear Loss\tVal Clear Accuracy\n')

def log_epoch(path, epoch, results):
    with open(path, 'a') as f:
        f.write(f'{epoch}\t' + '\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in results) + '\n')

def log_test(path, accuracy):
    with open(path, 'a') as f:
        f.write(f'FINAL_TEST\t-\t-\t-\t-\t-\t-\t{accuracy:.2f}\n')

