### Quantized Low-Rank Adaptation (QLoRA)

This project uses **QLoRA (Quantized Low-Rank Adaptation)** to fine-tune a pre-trained BERT model. QLoRA is an extension of LoRA that drastically reduces memory usage by quantizing the frozen pre-trained weights to 4-bit precision, while keeping the trainable LoRA adapters in higher precision (e.g., 16-bit float).

Mathematically, the standard LoRA forward pass $Y = X(W + \frac{\alpha}{r}BA)$ is modified by quantizing the base weights $W$ into a specialized 4-bit NormalFloat (NF4) format:

$$
Y = X(W_{\text{NF4}} + \frac{\alpha}{r}BA)
$$

During the backward pass, gradients are computed through the 4-bit weights and only applied to the small, 16-bit adapter matrices $A$ and $B$. This allows us to train massive models on consumer hardware without sacrificing performance.

---

# 📌 Checkpoint 1: Foundations for QLoRA Fine-Tuning

### 🔹 What is QLoRA?
QLoRA (Quantized Low-Rank Adaptation) takes LoRA a step further by quantizing the base model (e.g., to 4-bit precision) while fine-tuning small, low-rank adapters in a higher precision. This drastically reduces GPU memory usage, allowing massive models to be trained on consumer hardware.

### 🔹 BitsAndBytes
A library that enables seamless 8-bit and 4-bit quantization in Hugging Face models.
- Drastically reduces model memory footprint.
- **Normal Float 4 (nf4):** A specialized data type optimized for normally distributed weights, retaining higher accuracy than standard 4-bit floats.
- **Double Quantization:** Quantizes the quantization constants themselves for even more memory savings.

### 🔹 PEFT (Parameter-Efficient Fine-Tuning)
As with LoRA, PEFT wraps our quantized base model and attaches trainable low-rank matrices. Only these adapter weights are updated during training, ensuring efficiency and preserving the quantized backbone.

---

# 📌 Checkpoint 2: Applying QLoRA to BERT for Sequence Classification

### 🔹 Goal
Fine-tune a pre-trained BERT model using QLoRA for topic classification on the AG News dataset. This checkpoint prepares the dataset and wraps the model with a 4-bit quantization config.

### 🔹 Dataset Preparation
- **Dataset Choice:** `fancyzhx/ag_news` (News topic classification).
- **Preprocessing:**
  - Tokenize text with `AutoTokenizer`.
  - Apply truncation and padding (`max_length=128`).
  - Rename `label` → `labels` (required by Hugging Face models).
  - Format dataset into PyTorch tensors (`input_ids`, `attention_mask`, `labels`).

### 🔹 Model Setup with Quantization (BitsAndBytes)
```python
from transformers.utils import quantization_config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["classifier"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    device_map="auto",
    quantization_config=bnb_config
)
```
**Explanation:**
- `load_in_4bit=True` → Base model is loaded in 4-bit precision.
- `bnb_4bit_compute_dtype=torch.float16` → Computations occur in fp16 for speed and stability.
- `bnb_4bit_quant_type="nf4"` → Optimal data type for weights.
- `device_map="auto"` → Automatically places model layers on available GPUs.
- `llm_int8_skip_modules=["classifier"]` → Prevents the randomly initialized classifier head from being quantized to 4-bit, which would cause shape mismatches and crashes during training.

### 🔹 Applying LoRA on top of QLoRA
```python
from peft import prepare_model_for_kbit_training

qlora_adapter_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, qlora_adapter_config)
```
👉 *This leaves the base model frozen in 4-bit. We use `prepare_model_for_kbit_training` to prep the quantized model for gradients, and `modules_to_save=["classifier"]` to ensure the classification head is trained in full precision alongside the LoRA adapters.*

---

# 📌 Checkpoint 3: Training the Model with QLoRA

### 🔹 Goal
Fine-tune the QLoRA adapters using Hugging Face's Trainer API, leveraging the memory savings to train efficiently.

### 🔹 Training Setup
- **Batch size = 8** → Small batch size to ensure stability.
- **Learning rate = 5e-5** → Appropriate learning rate for sequence classification adapters.
- **Epochs = 2** → A quick training run for demonstration.

### 🔹 Metrics
We load the standard accuracy metric using the `evaluate` library. The `compute_metrics` function converts logits into predicted labels and compares them to the true dataset labels after every epoch.

### 🔹 TrainingArguments & Trainer
```python
qlora_training_args = TrainingArguments(
    output_dir="./qlora_agnews",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=2,
    eval_strategy="epoch",   
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=qlora_training_args,   
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)
trainer.train()
```
- The `Trainer` effortlessly manages the 4-bit quantized backbone and updates the fp16 adapters. 

### 🔹 Saving the Adapter
```python
model.save_pretrained("./qlora_agnews_adapter")
print("QLoRA fine-tuning complete! Adapter saved at ./qlora_agnews_adapter")
```
- Only the small adapter weights are saved to disk, keeping your storage overhead tiny!

---

# 📌 Checkpoint 4: Testing the Model on Custom Sentences

### 🔹 Goal
Verify our fine-tuned QLoRA model works by feeding it unseen sentences related to the AG News topics.

### 🔹 Label Mapping
```python
class_labels = dataset["train"].features["labels"].names
```
- The `class_labels` map directly to AG News categories (e.g., World, Sports, Business, Sci/Tech).

### 🔹 Sample Sentences
```python
samples = [
    "The stock market closed at an all-time high today.",
    "The football team won the championship after a thrilling game.",
    "Scientists discovered a new exoplanet in the habitable zone.",
    "The president met with foreign leaders to discuss trade agreements."
]
```
- These custom sentences touch on Business, Sports, Sci/Tech, and World news to ensure the model generalizes across classes.

### 🔹 Inference
```python
inputs = tokenizer(samples, truncation=True, padding=True, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)
```
- We set the model to `eval()` and run inputs through the network with no gradients (`torch.no_grad()`).
- Because of `device_map="auto"`, the model is on the GPU. We move our tokenized `inputs` to the GPU (`to(device)`) to match.

### 🔹 Display Predictions
```python
for text, pred in zip(samples, preds):
    print(f"News: {text}\nPredicted Topic: {class_labels[pred]}\n")
```

✅ **Checkpoint 4 Takeaway:**
You have successfully loaded a model in 4-bit precision, attached LoRA adapters, trained it on a GPU with minimal memory overhead, and evaluated its predictions! This is the state-of-the-art workflow for large model training.
