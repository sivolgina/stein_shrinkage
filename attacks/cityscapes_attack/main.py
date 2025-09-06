import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import sklearn.metrics as skm
import torch.optim as optim
from model import *
from train import *
from evaluate import *
from utils import *
from stein_corrected_bn import *
from mean_corrected_bn import *
from lasso_ridge_bn import *
import sys

if len(sys.argv) > 1:
    BN_TYPE = sys.argv[1]
else:
    BN_TYPE = 'stein'

if len(sys.argv) > 2:
    ATTACK_TYPE = sys.argv[2]
else:
    ATTACK_TYPE = 'FGSM' 

print(ATTACK_TYPE) 

path_data = "/cityscapes"
print(f"GPU: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 19
batch_size = 16
num_workers = 4
# Adapted from https://github.com/meetshah1995/pytorch-semseg


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def add_subgaussian_noise(input_tensor, sigma=100/255):
    eps = 1e-6
    U = torch.clamp(torch.rand(input_tensor.shape), min=eps, max=1 - eps)
    noise = - sigma /np.sqrt(2) * torch.tan(torch.pi * U) 
    return torch.clamp(input_tensor + noise, 0, 1)

class cityscapesLoader(data.Dataset):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    # makes a dictionary with key:value. For example 0:[128, 64, 128]
    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        # which data split to use
        split="train",
        # transform function activation
        is_transform=True,
        # image_size to use in transform function
        img_size=(256, 512),
        use_noise=False,
        noise_sigma=100/255,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.use_noise = use_noise
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.is_transform = is_transform
        self.files = {}
        self.noise_sigma = noise_sigma
        # makes it: /raid11/cityscapes/ + leftImg8bit + train (as we named the split folder this)
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        
        # contains list of all pngs inside all different folders. Recursively iterates 
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        ]
        
        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        
        # for void_classes; useful for loss function
        self.ignore_index = 250
        
        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # path of image
        img_path = self.files[self.split][index].rstrip()
        
        # path of label
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # read image
        img = imageio.imread(img_path)
        # convert to numpy array
        img = np.array(img, dtype=np.uint8)

        # read label
        lbl = imageio.imread(lbl_path)
        # encode using encode_segmap function: 0...18 and 250
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        return img, lbl

    def transform(self, img, lbl):       
         # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        lbl = Image.fromarray(lbl)
        
        # Resize the image and label
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        # Convert PIL Image back to numpy array
        img = np.array(img)
        lbl = np.array(lbl)

        # Change to BGR
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32) / 255.0


        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        # Ensure label values are within range
        classes = np.unique(lbl)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.use_noise:
            img = add_subgaussian_noise(img, sigma=self.noise_sigma)

        return img, lbl
      
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # !! Comment in code had wrong informtion
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    
full_train_data = cityscapesLoader(
    root=path_data,
    split='train',
    is_transform=True,
    use_noise=False
)
train_len = int(0.8 * len(full_train_data))
val_len = len(full_train_data) - train_len
train_subset, val_subset_base = random_split(full_train_data, [train_len, val_len])
val_subset_clean = Subset(full_train_data, val_subset_base.indices)


test_data = cityscapesLoader(
    root=path_data,
    split='val',
    is_transform=True,
    use_noise=False,
)


train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader_clean = DataLoader(val_subset_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_metrics(gt_label, pred_label):
    #Accuracy Score
    acc = skm.accuracy_score(gt_label, pred_label, normalize=True)
    
    #Jaccard Score/IoU
    js = skm.jaccard_score(gt_label, pred_label, average='micro')
    
    result_gm_sh = [acc, js]
    return(result_gm_sh)


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        # confusion matrix
        hist = self.confusion_matrix

        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis = 1) - hist.sum(axis = 0) + np.diag(hist)
        FP = hist.sum(axis = 1) - TP
        FN = hist.sum(axis = 0) - TP
    
        
        # Senstivity/Recall: TP / TP + FN
        sensti_cls = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        # Precision: TP / (TP + FP)
        prec_cls = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
        # F1 = 2 * Precision * Recall / Precision + Recall
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)
        
        return (
            {
                "Precision": prec,
                "Sensitivity": sensti,
                "F1": f1,
            }
        )
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

file_path = "logs/res.txt"
init_log_file(file_path)
num_runs=1

for _ in range(num_runs):
    model = get_seg_model(config)
    if BN_TYPE == 'stein':
        model = convert_batchnorm_stein(model).to(device)
    elif BN_TYPE == 'lasso':
        model = convert_batchnorm_lasso(model).to(device)
    elif BN_TYPE == 'ridge':
        model = convert_batchnorm_ridge(model).to(device)
    elif BN_TYPE == 'mean':
        model = convert_batchnorm_mean(model).to(device)
    elif BN_TYPE == 'vanilla':
        model = model.to(device)
    else:
        raise ValueError(f"Unknown BN type: {BN_TYPE}")

    #specify epsilon
    epsilon=0.03
    
    train_epochs = 1
    lr = 5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    best_js_clean = -float('inf')
    patience = 5
    epochs_without_improvement = 0
    best_model_state = None

    for epoch_i in range(train_epochs):
        print(f"Epoch {epoch_i + 1}\n-------------------------------")
        train_loss = train(train_loader, model, optimizer, epoch_i, train_epochs)
        val_clean = validate(val_loader_clean, model)

        results = [
            train_loss,
            val_clean["Precision"], val_clean["Sensitivity"], val_clean["F1"], val_clean["js"]
        ]

        log_epoch(file_path, epoch_i + 1, results)

        # Early stopping
        js_clean = val_clean["js"]
        if js_clean > best_js_clean:
            best_js_clean = js_clean
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch_i + 1}.')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if ATTACK_TYPE == "FGSM":
        test_attack_metrics = validate_with_fgsm_attack(model, test_loader, device, epsilon=epsilon)
    elif ATTACK_TYPE == "PGD":
        test_attack_metrics = validate_with_pgd_attack(model, test_loader, device, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown ATTACK_TYPE: {ATTACK_TYPE}")
    
    log_test(file_path, epsilon, test_attack_metrics)





