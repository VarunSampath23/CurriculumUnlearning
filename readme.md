# Uncertainty-Aware Curriculum for Machine Unlearning

**Controlling Path Dependence in Gradient Ascent Unlearning through Forget-Set Ordering**

This codebase aims to show that **the order in which you unlearn data matters** — and that using **MC-Dropout predictive entropy** as a curriculum signal can significantly reduce collateral damage on the retain set.

### Key Finding
When performing gradient ascent unlearning on an entire class:

- **Low-uncertainty-first** ordering → **+5.97%** retain accuracy over random
- **Low-uncertainty-first** also achieves the **best privacy** (lowest MIA score)

| Ordering                  | Forget Test Acc | Retain Test Acc | MIA Attack Acc |
|---------------------------|-----------------|-----------------|----------------|
| Random                    | 0.0%            | 79.00%          | 0.227          |
| **Low→High (Ours)**       | **0.0%**        | **84.97%**      | **0.172**      |
| High→Low                  | 0.0%            | 76.03%          | 0.247          |

*Base model (ImageNet-pretrained ResNet18 fine-tuned 5 epochs): 94.50% test accuracy*

---

## Quick Start
### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/cifar-unlearning-curriculum.git
cd cifar-unlearning-curriculum
pip install -r requirements.txt
### 1. Clone & Setup
```

### 2. Run Everything (Training + 3 Unlearning Experiments)
```bash
Bashchmod +x run_experiments.sh
./run_experiments.sh
```

This will:
Train the base ResNet18 for 5 epochs (saves checkpoint)
Run unlearning with random, low-uncertainty-first, and high-uncertainty-first orderings


### Citation
If you use this library or the findings of the paper in your work, please cite:

```
@inproceedings{makroosalsa,
  title={Controlling Path Dependence in Gradient Ascent Unlearning through Forget-Set Ordering},
  author={Sampath Kumar, Varun and Nadimi, Esmaeil S and Gogineni, Vinay Chakravarthi},
  booktitle={The 42st Conference on Uncertainty in Artificial Intelligence}
}
```
