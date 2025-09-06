import torch
import sklearn.metrics as skm
import numpy as np
from train import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        sensti_cls = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_cls)
        
        # Precision: TP / (TP + FP)
        prec_cls = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_cls)
        
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


def validate(val_loader, model):
    model.eval()
    running_metrics_val = runningScore(19)
    acc_sh = []
    js_sh = []
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            # Model prediction
            val_pred = model(val_images)
            pred = val_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()
            
            # Updating Mertics
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])
                                             
    score = running_metrics_val.get_scores()
    running_metrics_val.reset()

    # score["acc"] = np.mean(acc_sh)
    score["js"] = np.mean(js_sh)
    return score

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def validate_with_fgsm_attack(model, dataloader, device, epsilon=0.03):
    model.eval()
    running_metrics_val = runningScore(19)
    # empty list to add Accuracy and Jaccard Score Calculations
    acc_sh = []
    js_sh = []
    for val_images, val_labels in dataloader:
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)
        val_images.requires_grad = True
    
        val_pred = model(val_images)
        loss = cross_entropy2d(val_pred, val_labels)
        model.zero_grad()
        loss.backward()
        data_grad = val_images.grad.data
        perturbed_images = fgsm_attack(val_images, epsilon, data_grad)
        with torch.no_grad():
            perturbed_pred = model(perturbed_images)
            pred = perturbed_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])
                                             
    score = running_metrics_val.get_scores()
    running_metrics_val.reset()

    # score["acc"] = np.mean(acc_sh)
    score["js"] = np.mean(js_sh)
     
    return score

def pgd_attack(model, images, labels, device, epsilon=0.03, iters=5):
    ori_images = images.clone().detach().to(device)
    adv_images = ori_images.clone().detach()

    # adaptive step size
    alpha = epsilon / iters

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = cross_entropy2d(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.data
        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images


def validate_with_pgd_attack(model, dataloader, device, epsilon=0.03):
    model.eval()
    running_metrics_val = runningScore(19)
    acc_sh = []
    js_sh = []

    for val_images, val_labels in dataloader:
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        perturbed_images = pgd_attack(model, val_images, val_labels, device, epsilon=epsilon)
        with torch.no_grad():
            perturbed_pred = model(perturbed_images)
            pred = perturbed_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()

            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])

    score = running_metrics_val.get_scores()
    running_metrics_val.reset()
    score["js"] = np.mean(js_sh)

    return score
