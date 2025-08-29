# Knowledge Distillation for Classification Models
This project implements a **Knowledge Distillation** pipeline for training classification models.
The goal is to transfer knowledge from a **Teacher Model** (large and accurate) to a **Student Model** (smaller and faster), maintaining acceptable performance while reducing computational complexity.

Google Colab Notebook: [open here](https://colab.research.google.com/drive/1Q9wX0s9GstdoBj3Q8FJyi6BjlqOXbEqG?usp=sharing)

---

## ⚙️ Project Workflow

The notebook includes the following steps:

1. **Imports and Setup**

   * Install and import main libraries (`torch`, `torchvision`, etc.).

2. **Dataset Loading**

   * Use standard computer vision datasets (e.g., CIFAR-10 or similar).
   * Normalization and creation of `DataLoader`.

3. **Model Definition**

   * **Teacher Model**: a larger, more accurate network.
   * **Student Model**: a smaller, lighter network.

4. **Loss Functions**

   * **CrossEntropy Loss**: for classic supervised learning.
   * **Kullback–Leibler Divergence (KLDivLoss)**: for distillation from soft labels.
   * Combination of the two with a balancing parameter α (alpha).

5. **Training**

   * Step 1: Train the **teacher model** (if not pre-trained).
   * Step 2: Train the **student model** using both true labels and teacher predictions.

6. **Evaluation**

   * Compare teacher vs. student accuracy and performance.
   * Analyze parameter reduction and runtime improvements.

---

## 📊 Expected Results

* **Teacher model** → high accuracy but computationally heavy.
* **Student model** → lower accuracy, but much lighter.
* Proper tuning of α and temperature `T` allows the student to approach teacher-level performance while reducing computational cost.

---

## 📚 References
* Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*.
* Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. *Fitnets: Hints for thin deep nets*.
