# 📌 Checkpoint 1: Foundations for LoRA Fine-Tuning

### 🔹 Hugging Face
An AI ecosystem providing pre-trained models, datasets, and training utilities.

**Key hubs:**
- **Model Hub** → thousands of ready-to-use models.
- **Datasets Hub** → standardized datasets for NLP, vision, audio.
- **Trainer API** → simplifies training loops.

Makes fine-tuning reproducible and community-driven.

### 🔹 Transformers
A library by Hugging Face for state-of-the-art NLP models.
- Provides easy access to architectures like BERT, GPT, RoBERTa, LLaMA.
- Handles tokenization, model loading, and training seamlessly.

### 🔹 AutoModelForSequenceClassification
A pre-trained model class for text classification tasks.
- **Example:** Sentiment analysis (positive/negative).
- Automatically picks the right architecture based on the model name (e.g., BERT, DistilBERT).

### 🔹 AutoTokenizer
Converts raw text into tokens that the model understands.
- Ensures input matches the model's vocabulary.
- **Example:** `"LoRA is efficient"` → `[5023, 318, 1234]`.

### 🔹 TrainingArguments
A configuration object for training.

**Defines hyperparameters like:**
- Learning rate
- Batch size
- Number of epochs
- Logging and evaluation strategy

Keeps training setup organized and reproducible.

### 🔹 Trainer
Hugging Face's high-level training loop.

**Handles:**
- Training
- Evaluation
- Metrics
- Saving checkpoints

Saves you from writing boilerplate PyTorch code.

### 🔹 Pipeline
A ready-to-use abstraction for inference.

**Example:**
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
sentiment("LoRA is efficient")
```
Quickly applies fine-tuned models to tasks like sentiment analysis, translation, summarization.

### 🔹 PEFT (Parameter-Efficient Fine-Tuning)
A Hugging Face library for efficient fine-tuning methods.
- Includes LoRA, Prefix Tuning, Adapters.
- Lets you fine-tune huge models without updating all parameters.

### 🔹 LoRAConfig & get_peft_model
- **LoRAConfig:** Defines LoRA hyperparameters (rank, alpha, dropout).
- **get_peft_model:** Wraps the base model with LoRA adapters.

Ensures only LoRA parameters are trainable, keeping the backbone frozen.

### 🔹 Torch
**PyTorch** → the deep learning framework powering Hugging Face models.
- Handles tensor operations, GPU acceleration, and training loops.
- Hugging Face builds on top of PyTorch for flexibility.

### 🔹 Evaluate
Hugging Face library for metrics and evaluation.
- Provides standard metrics like accuracy, F1, perplexity.
- Integrates with Trainer for automatic evaluation during training.

---

# 📌 Checkpoint 2: Applying LoRA to BERT for Sequence Classification

### 🔹 Goal
Fine-tune a pre-trained BERT model using LoRA (Low-Rank Adaptation) for a multi-class text classification task.
This checkpoint demonstrates how to prepare the dataset, tokenize inputs, and wrap the model with LoRA adapters.

### 🔹 Dataset Preparation
- **Dataset Choice:** Example → IMDb (sentiment analysis) or AG News (topic classification).

**Preprocessing:**
- Tokenize text with `AutoTokenizer`.
- Apply truncation and padding (`max_length=128`).
- Rename `label` → `labels` (required by Hugging Face models).
- Format dataset into PyTorch tensors (`input_ids`, `attention_mask`, `labels`).

👉 *This ensures the dataset is ready for training with BERT.*

### 🔹 Model Setup
```python
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
```
- Loads BERT (uncased) → ignores capitalization.
- Adds a classification head for 6 labels (multi-class classification).
- **Example:** News categories (World, Sports, Business, Sci/Tech, Entertainment, Politics).

### 🔹 Applying LoRA
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)
```
**Explanation:**
- `r=8` → Rank of low-rank matrices (controls trainable parameter size).
- `lora_alpha=16` → Scaling factor for updates.
- `target_modules=["query", "value"]` → LoRA applied only to attention layers (efficient adaptation).
- `lora_dropout=0.1` → Prevents overfitting.
- `task_type="SEQ_CLS"` → Sequence classification task.
- `get_peft_model` → Wraps BERT with LoRA adapters so only LoRA parameters are trained.

👉 *This makes fine-tuning lightweight and efficient.*

### 🔹 Training Setup
- **TrainingArguments** → Defines hyperparameters (learning rate, batch size, epochs).
- **Trainer** → Handles training loop, evaluation, and saving checkpoints.
- **Evaluate** → Provides metrics (accuracy, F1 score, etc.).

### 🔹 Why This Matters
- **Efficiency:** Only LoRA parameters are trained, not the full BERT model.
- **Reusability:** Base model remains intact for other tasks.
- **Portability:** LoRA weights are small and easy to share.
