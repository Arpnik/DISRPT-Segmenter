import argparse

import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import wandb

from com.disrpt.segmenter.dataset_prep import download_gum_dataset, load_gum_datasets
import warnings
warnings.filterwarnings("ignore")

class BERTFineTuning:
    """
    Enhanced BERT fine-tuning with configurable checkpointing, 
    early stopping, and comprehensive evaluation.
    """

    def __init__(self, model_name, num_labels=2, device='cuda'):
        """
        Initialize BERT model with LoRA for EDU segmentation.

        Args:
            model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased')
            num_labels: Number of classes (2 for binary EDU segmentation)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels

        print("\n" + "=" * 70)
        print(f"Initializing {model_name} with LoRA")
        print("=" * 70)

        # Load base model
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"] if "distilbert" in model_name.lower()
            else ["query", "value"],
            bias="none"
        )

        # Apply LoRA
        self.model = get_peft_model(model, lora_config)
        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()

    @staticmethod
    def compute_metrics(pred):
        """
        Compute precision, recall, F1, and accuracy for EDU segmentation.

        Args:
            pred: Tuple of (predictions, labels) from Trainer

        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        # Flatten and filter out ignored tokens (-100)
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

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_model(
            self,
            train_dataset,
            eval_dataset,
            output_dir,
            num_epochs=10,
            batch_size=16,
            eval_batch_size=32,
            learning_rate=3e-4,
            save_every_n_epochs=2,
            early_stopping_patience=3,
            early_stopping_threshold=0.001
    ):
        """
        Train model with configurable checkpointing and early stopping.

        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset
            output_dir: Directory to save model and checkpoints
            num_epochs: Total training epochs
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            learning_rate: Learning rate for optimizer
            save_every_n_epochs: Save checkpoint every N epochs
            early_stopping_patience: Epochs to wait before stopping
            early_stopping_threshold: Minimum improvement threshold

        Returns:
            Dictionary with final evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model:                    {self.model_name}")
        print(f"Output directory:         {output_dir}")
        print(f"Training examples:        {len(train_dataset)}")
        print(f"Validation examples:      {len(eval_dataset)}")
        print(f"Epochs:                   {num_epochs}")
        print(f"Batch size:               {batch_size}")
        print(f"Learning rate:            {learning_rate}")
        print(f"Save every N epochs:      {save_every_n_epochs}")
        print(f"Early stopping patience:  {early_stopping_patience}")
        print(f"Device:                   {self.device}")
        print("=" * 70)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,

            # Logging
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            logging_strategy="steps",

            # Evaluation
            eval_strategy="epoch",

            # Checkpointing - save every N epochs
            save_strategy="epoch",
            save_steps=save_every_n_epochs,
            save_total_limit=5,  # Keep only last 5 checkpoints

            # Best model selection
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,

            # Disable external logging
            report_to="wandb" if wandb.run is not None else "none",

        )

        # Data collator for dynamic padding
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )

        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )

        # Train
        print("\n" + "🚀" * 35)
        print("TRAINING STARTED")
        print("🚀" * 35 + "\n")

        train_result = trainer.train()

        print("\n" + "✅" * 35)
        print("TRAINING COMPLETED")
        print("✅" * 35)

        # Print training metrics
        print("\n" + "=" * 70)
        print("FINAL TRAINING METRICS")
        print("=" * 70)
        for key, value in train_result.metrics.items():
            print(f"{key:.<50} {value:.4f}")

        # Evaluate on validation set
        print("\n" + "=" * 70)
        print("VALIDATION METRICS")
        print("=" * 70)

        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"{key:.<50} {value:.4f}")

        # Save final model
        final_model_dir = output_path / "best_model"
        print("\n" + "=" * 70)
        print(f"Saving best model to: {final_model_dir}")
        print("=" * 70)

        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        print("✓ Model saved successfully")

        return eval_results

    def evaluate_test_set(self, test_dataset, model_path, batch_size=32):
        """
        Load trained model and evaluate on test set.

        Args:
            test_dataset: Test dataset
            model_path: Path to saved model directory
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with test metrics and detailed classification report
        """
        print("\n" + "=" * 70)
        print("LOADING MODEL FOR TEST EVALUATION")
        print("=" * 70)
        print(f"Model path: {model_path}")
        print(f"Test examples: {len(test_dataset)}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels
        )
        model.to(self.device)

        print("✓ Model loaded successfully")

        # Create data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )

        # Training args for evaluation only
        eval_args = TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),
            report_to="wandb" if wandb.run is not None else "none"
        )

        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Evaluate on test set
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)

        test_results = trainer.evaluate(test_dataset)

        # Print metrics
        print("\n📊 TEST METRICS:")
        print("-" * 70)
        for key, value in test_results.items():
            print(f"{key:.<50} {value:.4f}")

        # Get detailed predictions for classification report
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

        # Print classification report
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

        return test_results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for EDU segmentation with LoRA + W&B")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./output/edu_segmenter")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="edu-segmentation")
    parser.add_argument("--wandb_run_name", type=str, default="")

    return parser.parse_args()


def main():
    """Complete training pipeline with train/dev/test evaluation"""

    print("\n" + "=" * 35)
    print("EDU SEGMENTATION MODEL TRAINING")
    print("=" * 35)

    # Configuration
    args = parse_args()
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs2
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    LEARNING_RATE = args.learning_rate

    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project= args.wandb_project,
            name= args.wandb_run_name if args.wandb_run_name!="" else f"{MODEL_NAME.replace('/', '-')}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}",
            config={
                "model_name": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            }
        )
    # Step 1: Download dataset
    print("\n" + "=" * 70)
    print("STEP 1: Download Dataset")
    print("=" * 70)
    download_success = download_gum_dataset()

    if not download_success:
        print("\n❌ Dataset download failed!")
        return

    # Step 2: Load tokenizer and datasets
    print("\n" + "=" * 70)
    print("STEP 2: Load Datasets")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, dev_dataset, test_dataset = load_gum_datasets(tokenizer)

    # Step 3: Initialize model
    print("\n" + "=" * 70)
    print("STEP 3: Initialize Model")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    bert_model = BERTFineTuning(
        model_name=MODEL_NAME,
        num_labels=2,
        device=device
    )

    # Step 4: Train model
    print("\n" + "=" * 70)
    print("STEP 4: Train Model")
    print("=" * 70)

    eval_results = bert_model.train_model(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        learning_rate=LEARNING_RATE
    )

    # Step 5: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 5: Final Test Set Evaluation")
    print("=" * 70)

    best_model_path = Path(OUTPUT_DIR) / "best_model"
    test_results = bert_model.evaluate_test_set(
        test_dataset=test_dataset,
        model_path=str(best_model_path)
    )

    # Final summary
    print("\n" + "🎉" * 35)
    print("TRAINING PIPELINE COMPLETE!")
    print("🎉" * 35)

    print("\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 70)
    print(f"{'Metric':<30} {'Validation':<20} {'Test':<20}")
    print("-" * 70)

    metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
    for metric in metrics:
        metric_name = metric.replace('eval_', '').capitalize()
        val_score = eval_results.get(metric, 0)
        test_score = test_results.get(metric, 0)
        print(f"{metric_name:<30} {val_score:<20.4f} {test_score:<20.4f}")

    print("=" * 70)
    print(f"\n✅ Best model saved at: {best_model_path}")
    print(f"✅ Training logs saved at: {OUTPUT_DIR}/logs")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
