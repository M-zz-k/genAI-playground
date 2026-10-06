### Low-Rank Adaptation (LoRA)

This project uses **LoRA (Low-Rank Adaptation)** to fine-tune a pretrained BERT model for **emotion classification** without updating all of BERT's parameters.

LoRA keeps the original BERT weights **frozen** and adds small trainable low-rank matrices to selected attention layers. In this implementation, LoRA is applied to the **Query and Value** projections using `r=8` and `lora_alpha=16`. Instead of directly learning a large weight update, LoRA represents it as a product of smaller matrices:

$$
W' = W + \frac{\alpha}{r}BA
$$

where `r` controls the rank/capacity of the adapter and `alpha` controls the scaling of the LoRA update.

During training, **only the LoRA parameters are updated**, significantly reducing the number of trainable parameters, memory usage, and computational cost compared with full fine-tuning. The adapted model is then evaluated on unseen test data using classification metrics such as **accuracy, precision, recall, and F1-score** to verify that the model retains good performance while being parameter-efficient.

**In short:** LoRA adapts a large pretrained model by **freezing the original model and training a small task-specific adapter**, providing an efficient alternative to full fine-tuning.

---

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

---

# 📌 Checkpoint 3: Training the Model with LoRA

### 🔹 Goal
Fine-tune only the LoRA layers of BERT using Hugging Face's Trainer API.
We keep training lightweight and efficient by updating only LoRA parameters.

### 🔹 Training Setup
- **Batch size = 8** → Small batch size to fit into limited GPU memory.
- **Learning rate = 2e-4** → Slightly higher since only LoRA layers are trained.
- **Epochs = 2** → Short training for demonstration; more epochs improve accuracy.

### 🔹 Metrics
```python
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)
```
- `evaluate.load("accuracy")` → Loads accuracy metric.
- `compute_metrics` → Converts model outputs (logits) into predicted labels and compares with ground truth.

Ensures evaluation is automatic during training.

### 🔹 TrainingArguments
```python
args = TrainingArguments(
    output_dir="./lora_emotion",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=2,    
    eval_strategy="epoch",
    logging_steps=10,
    report_to="none"
)
```
**Key Parameters:**
- `output_dir` → Where checkpoints and logs are saved.
- `per_device_train_batch_size` → Training batch size per GPU.
- `per_device_eval_batch_size` → Evaluation batch size.
- `learning_rate` → Controls update speed of LoRA parameters.
- `num_train_epochs` → Number of passes over dataset.
- `eval_strategy="epoch"` → Evaluate after each epoch.
- `logging_steps=10` → Log progress every 10 steps.
- `report_to="none"` → Disables external logging (e.g., WandB).

### 🔹 Trainer
```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)
trainer.train()
```
- **Trainer** → High-level API that handles training loop, evaluation, and logging.
- `train_dataset` → Training split.
- `eval_dataset` → Validation split.
- `compute_metrics` → Evaluates accuracy after each epoch.
- `trainer.train()` → Starts fine-tuning process.

### 🔹 Saving the Adapter
```python
model.save_pretrained("./lora_emotion_adapter")
print("LoRA fine-tuning complete! Adapter saved at ./lora_emotion_adapter")
```
- Saves only LoRA adapter weights, not the full BERT model.
- Adapter can be reloaded later on top of the base model.
- Lightweight and portable for sharing or deployment.

### 🔹 Output & Results
**Example run:** Accuracy ≈ 54% after 2 epochs on a small subset.

Low accuracy is expected because:
- Few epochs.
- Small dataset subset.

**For better performance:**
- Train longer (5–10 epochs).
- Use full dataset.
- Tune hyperparameters.

---

# 📌 Checkpoint 4: Testing the Model on Custom Sentences

### 🔹 Goal
Verify that the LoRA-fine-tuned BERT model works correctly by running it on custom sentences.
This confirms that the LoRA adapter is applied and the model can classify emotions.

### 🔹 Device Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
- Uses GPU (CUDA) if available, otherwise falls back to CPU.
- Moves the model to the chosen device for inference.

### 🔹 Emotion Labels
```python
emotion_labels = dataset["train"].features["labels"].names
```
- Retrieves the class names from the dataset.
- **Example:** `["joy", "fear", "surprise", "sadness", "love", "anger"]`.

### 🔹 Sample Sentences
```python
samples = [
    "I am so happy to see you!",
    "This is terrifying, I can't handle it.",
    "He surprised everyone with his gift.",
    "I feel so sad and lonely.",
    "I love spending time with my family."
]
```
- Custom test inputs to check model predictions.
- Covers different emotions: joy, fear, surprise, sadness, love.

### 🔹 Tokenization
```python
inputs = tokenizer(samples, truncation=True, padding=True, return_tensors="pt").to(device)
```
- Converts sentences into token IDs and attention masks.
- Pads/truncates to uniform length.
- Returns PyTorch tensors ready for inference.

### 🔹 Model Inference
```python
model.eval()
with torch.no_grad():
    logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)
```
- `model.eval()` → Sets model to evaluation mode (no dropout).
- `torch.no_grad()` → Disables gradient tracking (faster inference).
- `logits` → Raw model outputs (unnormalized scores).
- `torch.argmax` → Picks the highest-scoring label for each sentence.

### 🔹 Display Predictions
```python
for text, pred in zip(samples, preds):
    print(f"Text: {text}\nPredicted Emotion: {emotion_labels[pred]}\n")
```
Loops through sentences and prints predicted emotion.

**Example output:**
```text
Text: I am so happy to see you!
Predicted Emotion: joy
```

### 🔹 Output & Results
**Example run:** Accuracy may be modest (~54%) because:
- Trained only for 2 epochs.
- Used a small subset of dataset.

**For better performance:**
- Train longer (5–10 epochs).
- Use full dataset.
- Tune hyperparameters.

✅ **Checkpoint 4 Takeaway:**
You can now test your LoRA-fine-tuned model on custom sentences. This confirms the adapter works, the pipeline is complete, and you can showcase results in your portfolio.
