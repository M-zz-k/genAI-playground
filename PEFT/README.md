### Parameter-Efficient Fine-Tuning (PEFT)

This project explores the use of **PEFT (Parameter-Efficient Fine-Tuning)** with the Hugging Face `peft` library. PEFT techniques, like LoRA, freeze the large pre-trained base model and inject small, trainable modules (adapters) into the architecture. 

Mathematically, this allows us to compute weight updates $W' = W + \Delta W$ where $\Delta W$ is constrained to a much smaller rank or subset of parameters. This dramatically reduces the number of trainable parameters, memory footprint, and training time, while achieving performance comparable to full fine-tuning. 

In this experiment, we apply PEFT via LoRA to a `bert-base-uncased` model for sentiment analysis on the IMDB dataset.

---

# 📌 Checkpoint 1: Foundations for PEFT

### 🔹 What is PEFT?
The Hugging Face `peft` library provides an elegant interface to apply state-of-the-art parameter-efficient fine-tuning methods (like LoRA, Prefix Tuning, P-Tuning) to pre-trained models.

### 🔹 Setting up the Environment
We import the necessary libraries: `datasets`, `torch`, `transformers`, and `peft`. We configure our device to prioritize CUDA if available, ensuring hardware acceleration.

---

# 📌 Checkpoint 2: Data Preparation

### 🔹 Dataset Choice
- **Dataset:** `stanfordnlp/imdb` (Movie Reviews).
- **Goal:** Sentiment classification (Binary: Positive or Negative).

### 🔹 Preprocessing
- **Tokenization:** Using `AutoTokenizer` for `bert-base-uncased`.
- **Formatting:** We apply `truncation=True` and `padding="max_length"` to length 128.
- **Conversion:** We rename the target column from `label` to `labels` (which the `Trainer` expects) and convert the entire dataset to PyTorch tensors via `encoded_dataset.set_format("torch")`.

---

# 📌 Checkpoint 3: Injecting the PEFT Adapters

### 🔹 Loading the Base Model
We load `bert-base-uncased` with `AutoModelForSequenceClassification` configured for 2 labels.

### 🔹 Configuring LoRA via PEFT
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
```
- We define a LoRA configuration targeting the attention `query` and `value` matrices.
- Rank (`r=8`) constraints the adapter size.
- `task_type="SEQ_CLS"` tells PEFT we are doing Sequence Classification.

### 🔹 Wrapping the Model
```python
model = get_peft_model(model, lora_config)
```
- `get_peft_model` magically wraps the base BERT model. It freezes the backbone and injects the new trainable adapter layers. Only the adapters will compute gradients.

---

# 📌 Checkpoint 4: Training with the Trainer API

### 🔹 Training Setup
We use the Hugging Face `Trainer` API, configuring `TrainingArguments` to store outputs in `./lora-imdb`.
- **Batch Size:** 16 (for both training and evaluation).
- **Learning Rate:** 2e-4 (PEFT models often require slightly higher learning rates than full fine-tuning).
- **Epochs:** 2.

### 🔹 Execution
```python
trainer.train()
```
- Because we are using PEFT, training is incredibly fast and memory-efficient.

### 🔹 Saving
```python
model.save_pretrained("./lora-imdb-adapter")
```
- When we save the model, `peft` intelligently saves **only the small adapter weights** (usually just a few megabytes), not the entire base model.

---

# 📌 Checkpoint 5: Inference Pipeline

### 🔹 Loading the Fine-Tuned Model
We leverage the Hugging Face `pipeline` for easy inference. We point the `model` argument to our saved `./lora-imdb-adapter`. The pipeline handles loading the base model and automatically attaching the saved LoRA adapters.

### 🔹 Testing the Model
```python
sentiment = pipeline(
    "text-classification",
    model="./lora-imdb-adapter",
    tokenizer="bert-base-uncased"
)
```
We provide sample reviews:
1. *"I absolutely loved this film—best sci-fi I've seen in years!"* → **POSITIVE**
2. *"It was okay, not great, but worth a watch."* → **POSITIVE** (or leaning neutral)
3. *"Terrible plot, terrible acting, total waste of time."* → **NEGATIVE**

✅ **Takeaway:** Using the `peft` library, we easily isolated the trainable parameters to a tiny fraction of the base model, achieved fast training, and maintained high predictive accuracy on unseen data!
