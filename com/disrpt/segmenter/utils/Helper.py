import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from transformers import TrainingArguments, Trainer
import wandb
import torch

def _get_device() -> str:
    """Automatically detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        # print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        # print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        # print("Using CPU")
    return device

def compute_metrics(pred):
    """
    Compute precision, recall, F1, and accuracy.
    Logs to W&B if active.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter out -100 labels
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for pred_label, true_label in zip(prediction, label):
            if true_label != -100:
                true_predictions.append(pred_label)
                true_labels.append(true_label)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        true_predictions,
        average='binary',
        pos_label=1,
        zero_division=0
    )
    acc = accuracy_score(true_labels, true_predictions)

    # Also calculate per-class metrics for W&B
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels,
        true_predictions,
        average=None,
        zero_division=0
    )

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_class_0': precision_per_class[0],
        'precision_class_1': precision_per_class[1],
        'recall_class_0': recall_per_class[0],
        'recall_class_1': recall_per_class[1],
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'support_class_0': int(support[0]),
        'support_class_1': int(support[1])
    }

    return metrics


def evaluate_test_set(test_dataset, model_path, model, data_collator, batch_size=32, use_wandb=False):
    """
    Load trained model and evaluate on test set.
    Logs results to W&B.
    """
    print("\n" + "=" * 70)
    print("LOADING MODEL FOR TEST EVALUATION")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Test examples: {len(test_dataset)}")

    model = model.to(_get_device())
    model.eval()

    print("✓ Model loaded successfully")

    # Evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

        # Create trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    test_results = trainer.evaluate(test_dataset)

    if use_wandb:
        wandb.log({
            "test/loss": test_results.get("eval_loss", 0),
            "test/accuracy": test_results.get("eval_accuracy", 0),
            "test/precision": test_results.get("eval_precision", 0),
            "test/recall": test_results.get("eval_recall", 0),
        })
        # Print metrics
        print("\n📊 TEST METRICS:")
        print("-" * 70)
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                print(f"{key:.<50} {value:.4f}")

    # Log to W&B
    if use_wandb:
        wandb.log({f"test_{k}": v for k, v in test_results.items()})

    # Get predictions for classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=2)
    true_labels = predictions.label_ids

    # Flatten and filter
    flat_preds = []
    flat_labels = []
    for preds, labels in zip(pred_labels, true_labels):
        for pred, label in zip(preds, labels):
            if label != -100:
                flat_preds.append(pred)
                flat_labels.append(label)

    # Classification report
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    print("\nClass Labels:")
    print("  0 = EDU Continue (token within current EDU)")
    print("  1 = EDU Start (token begins new EDU)\n")

    report = classification_report(
        flat_labels,
        flat_preds,
        target_names=['EDU Continue (0)', 'EDU Start (1)'],
         digits=4
    )
    print(report)

    if use_wandb:

        cm = confusion_matrix(flat_labels, flat_preds)
        wandb.log({
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=flat_labels,
                preds=flat_preds,
                class_names=['Continue', 'Start']
            )
        })

    return test_results


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100
        )

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_dataset, class_1_multiplier=2.0):
    """
    Calculate class weights with optional amplification for class 1

    Args:
        class_1_multiplier: Multiply class 1 weight by this factor (>1 improves recall)
    """
    all_labels = []
    for item in train_dataset:
        labels = item['labels']
        valid_labels = [l for l in labels if l != -100]
        all_labels.extend(valid_labels)

    # Compute balanced weights first
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=np.array(all_labels)
    )

    # Amplify class 1 weight for better recall
    class_weights[1] *= class_1_multiplier

    print(f"\n⚖️ Adjusted Class Weights (multiplier={class_1_multiplier}):")
    print(f"   Label 0 (Continue): {class_weights[0]:.4f}")
    print(f"   Label 1 (Start):    {class_weights[1]:.4f}")
    print(f"   Ratio (1/0):        {class_weights[1] / class_weights[0]:.4f}x")

    return torch.tensor(class_weights, dtype=torch.float32)