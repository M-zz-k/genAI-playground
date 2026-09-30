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
