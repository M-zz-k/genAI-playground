### Knowledge Distillation (KD)

This project explores **Knowledge Distillation**, a model compression technique where a smaller, lightweight model (the **"Student"**) is trained to mimic the behavior of a larger, complex model (the **"Teacher"**).

Instead of only learning from the hard ground-truth labels (e.g., [1, 0, 0]), the student also learns from the "soft labels" (the output distribution) produced by the teacher. These soft labels contain dark knowledge—information about how the teacher evaluates incorrect classes, which helps the student generalize much better than training on hard labels alone.

Mathematically, the Knowledge Distillation loss is a weighted combination of two components:

1. **Student Loss (Hard Target):** Standard Cross-Entropy between the student's output and the true label.
2. **Distillation Loss (Soft Target):** Kullback-Leibler (KL) Divergence between the soft probabilities of the student and teacher, scaled by a **Temperature ($T$)**.

The total loss is:
$$
\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y_{\text{true}}, \sigma(Z_s)) + (1 - \alpha) T^2 \mathcal{L}_{\text{KL}} \left( \sigma\left(\frac{Z_s}{T}\right), \sigma\left(\frac{Z_t}{T}\right) \right)
$$

Where:
- $Z_s, Z_t$: Logits of the Student and Teacher.
- $\sigma$: Softmax function.
- $T$: Temperature. Higher $T$ creates a softer probability distribution over classes.
- $\alpha$: Weighting factor balancing the hard and soft losses.

---

# 📌 Checkpoint 1: Foundations & Defining the Models

### 🔹 Goal
Set up the PyTorch environment and define the neural network architectures for both the Teacher and the Student.

### 🔹 Steps
- **Teacher Model:** Define a large, high-capacity model (e.g., deep layers, wide hidden dimensions).
- **Student Model:** Define a much smaller, shallower model that will learn from the Teacher.
- **Initialization:** Instantiate both models. The Teacher is typically pre-trained, but for simple experimental setups, it can be trained from scratch first.

---

# 📌 Checkpoint 2: The Distillation Loss Function

### 🔹 Goal
Implement the custom loss function that combines both Cross-Entropy (Hard Loss) and KL-Divergence (Soft Loss).

### 🔹 Steps
- Apply standard `nn.CrossEntropyLoss` for the hard targets.
- Apply `nn.KLDivLoss` comparing the log-softmax of the student's scaled logits to the softmax of the teacher's scaled logits.
- Scale the KL Divergence by $T^2$ to ensure the gradients of both loss components are comparable in magnitude.

---

# 📌 Checkpoint 3: Training the Teacher (Optional Phase)

### 🔹 Goal
If the Teacher model isn't pre-trained, we must first train it on the dataset using standard Cross-Entropy loss so it can acquire knowledge.

### 🔹 Steps
- Set up a standard training loop (forward pass, loss, backward pass, optimizer step).
- Train the Teacher model until convergence.
- **Freeze the Teacher:** Once trained, set `teacher.eval()` and wrap its inference in `torch.no_grad()` to freeze its weights during the distillation phase.

---

# 📌 Checkpoint 4: Distilling Knowledge into the Student

### 🔹 Goal
Train the Student model using the Knowledge Distillation loss.

### 🔹 Steps
- Iterate through the training dataset.
- **Teacher Forward Pass:** Pass inputs through the frozen Teacher to get target logits ($Z_t$).
- **Student Forward Pass:** Pass inputs through the Student to get student logits ($Z_s$).
- **Compute Total Loss:** Pass $Z_t, Z_s$, and the true labels to your custom Distillation Loss function.
- **Backpropagation:** Update only the Student's parameters.

---

# 📌 Checkpoint 5: Evaluation & Comparison

### 🔹 Goal
Evaluate the success of the distillation by comparing the Student's performance to the Teacher's.

### 🔹 Steps
- Test both models on unseen data.
- **Metrics to Compare:**
  - **Accuracy:** Does the Student match (or come close to) the Teacher?
  - **Inference Time:** How much faster is the Student?
  - **Model Size:** Compare the parameter counts to highlight the compression ratio.

✅ **Takeaway:** A successful Knowledge Distillation experiment will yield a Student model that is significantly faster and smaller than the Teacher, yet performs much better than if it had been trained on the dataset alone without the Teacher's guidance!
