# stein_shrinkage
Code for the paper "Admissibility of Stein Shrinkage for Batch Normalization in the Presence of Adversarial Attacks".

## Structure
This repository contains two main folders:
* `attacks/` – experiments with data-dependent adversarial attacks
* `subgaussian_noise/` – experiments with sub-Gaussian noise
Inside each folder, select the dataset folder you want to run experiments on.

## Running Experiments

### Adversarial Attacks

Before `running main.py`, you can specify:
* BN version,
* attack type and its level,
* model,
* loss criterion,
* optimizer.
  
**Example:**
```bash
python main.py stein FGSM
```
Here: "stein" specifies the BN version, "FGSM" specifies the attack type.

### Sub-Gaussian Noise

Before `running main.py`, you can specify:
* BN version,
* model,
* loss criterion,
* optimizer.

**Example:**
```bash
python main.py stein 
```
Here: "stein" specifies the BN version

### Hyperparameters
All hyperparameters are listed in the Appendix (experiments section) of the paper.

### Requirements

We recommend Python 3.8+, install dependencies via `pip`:
```bash
pip install torch torchvision torchattacks numpy pandas scikit-learn Pillow imageio neuroCombat
```
