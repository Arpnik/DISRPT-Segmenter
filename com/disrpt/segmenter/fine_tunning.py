import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import wandb
from com.disrpt.segmenter.dataset_prep import download_dataset, load_datasets
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# MODEL ARCHITECTURE: BERT + LoRA + MLP Classifier
# ============================================================================

class BERTWithMLPClassifier(nn.Module):
    """
    BERT encoder with LoRA adapters + Multi-Layer Perceptron classifier head.

    Architecture:
        BERT (with LoRA) → MLP [768 → 256 → 128 → 2]

    Both LoRA parameters and MLP weights are trainable.
    """

    def __init__(
            self,
            model_name,
            num_labels=2,
            mlp_hidden_dims=[256, 128],
            mlp_dropout=0.3,
            activation='gelu'
    ):
        super(BERTWithMLPClassifier, self).__init__()

        # Load BERT encoder (without default classifier head)
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.config = config

        # Select activation function
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        self.activation = activations.get(activation.lower(), nn.GELU())

        # Build MLP classifier head
        layers = []
        input_dim = self.hidden_size

        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(mlp_dropout)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

        # Print architecture
        print(f"\n🏗️  MLP Classifier Architecture:")
        print(f"   Input:  {self.hidden_size} (BERT hidden size)")
        for i, dim in enumerate(mlp_hidden_dims):
            print(
                f"   Layer {i + 1}: Linear({input_dim if i == 0 else mlp_hidden_dims[i - 1]} → {dim}) → {activation.upper()} → Dropout({mlp_dropout})")
        print(f"   Output: Linear({mlp_hidden_dims[-1]} → {num_labels})")

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        """
        Forward pass with loss calculation.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Ground truth labels (batch_size, seq_length)
            inputs_embeds: Optional pre-computed embeddings
            **kwargs: Additional arguments (ignored)

        Returns:
            dict with 'loss' and 'logits'
        """
        # Get BERT encodings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )

        # Token-level hidden states: (batch_size, seq_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Pass through MLP classifier: (batch_size, seq_length, num_labels)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # CrossEntropyLoss automatically ignores -100 labels
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return {
            'loss': loss,
            'logits': logits
        }


# ============================================================================
# TRAINING CLASS
# ============================================================================

class BERTFineTuning:
    """
    BERT fine-tuning with LoRA + MLP classifier for EDU segmentation.
    Includes W&B logging, early stopping, and comprehensive evaluation.
    """

    def __init__(
            self,
            model_name,
            num_labels=2,
            device='cuda',
            mlp_hidden_dims=[256, 128],
            mlp_dropout=0.3,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of output classes (2 for EDU segmentation)
            device: Device for training
            mlp_hidden_dims: Hidden layer dimensions for MLP
            mlp_dropout: Dropout rate in MLP
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
        """
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels

        print("\n" + "=" * 70)
        print(f"Initializing {model_name} with LoRA + MLP Classifier")
        print("=" * 70)

        # Create model with MLP head
        model = BERTWithMLPClassifier(
            model_name=model_name,
            num_labels=num_labels,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_dropout=mlp_dropout,
            activation='gelu'
        )

        # Configure LoRA for BERT encoder only
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # We have custom head
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin"] if "distilbert" in model_name.lower()
            else ["query", "value"],
            bias="none",
            modules_to_save=["classifier"]  # Train MLP alongside LoRA
        )

        # Apply LoRA
        self.model = get_peft_model(model, lora_config)

        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()

        # Move to device
        self.model = self.model.to(device)

    @staticmethod
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
        Train model with LoRA + MLP classifier.
        Logs all metrics to W&B.
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
        print(f"W&B logging:              {'Enabled' if wandb.run is not None else 'Disabled'}")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            warmup_ratio=0.1,
            weight_decay=0.01,

            # Logging
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            logging_strategy="steps",
            logging_first_step=True,

            # Evaluation
            eval_strategy="epoch",
            eval_steps=None,

            # W&B integration
            report_to="wandb" if wandb.run is not None else "none",

            # Checkpointing
            save_strategy="epoch",
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            dataloader_pin_memory=True,

            # Gradient settings
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )

        # Early stopping
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

        # Log model architecture to W&B
        if wandb.run is not None:
            wandb.watch(self.model, log="all", log_freq=100)

        print("\n" + "🚀" * 35)
        print("TRAINING STARTED")
        print("🚀" * 35 + "\n")

        train_result = trainer.train()

        print("\n" + "✅" * 35)
        print("TRAINING COMPLETED")
        print("✅" * 35)

        # Print and log training metrics
        print("\n" + "=" * 70)
        print("FINAL TRAINING METRICS")
        print("=" * 70)
        for key, value in train_result.metrics.items():
            print(f"{key:.<50} {value:.4f}")

        if wandb.run is not None:
            wandb.log({f"train_final_{k}": v for k, v in train_result.metrics.items()})

        # Evaluate on validation set
        print("\n" + "=" * 70)
        print("VALIDATION METRICS")
        print("=" * 70)
        eval_results = trainer.evaluate()

        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"{key:.<50} {value:.4f}")

        if wandb.run is not None:
            wandb.log({f"eval_final_{k}": v for k, v in eval_results.items()})

        # Save best model
        final_model_dir = output_path / "best_model"
        print("\n" + "=" * 70)
        print(f"Saving best model to: {final_model_dir}")
        print("=" * 70)

        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        # Save model artifact to W&B
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"edu-segmenter-{wandb.run.id}",
                type="model",
                description="Best EDU segmentation model"
            )
            artifact.add_dir(str(final_model_dir))
            wandb.log_artifact(artifact)

        print("✓ Model saved successfully")

        return eval_results

    def evaluate_test_set(self, test_dataset, model_path, batch_size=32):
        """
        Load trained model and evaluate on test set.
        Logs results to W&B.
        """
        print("\n" + "=" * 70)
        print("LOADING MODEL FOR TEST EVALUATION")
        print("=" * 70)
        print(f"Model path: {model_path}")
        print(f"Test examples: {len(test_dataset)}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model architecture
        base_model = BERTWithMLPClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            mlp_hidden_dims=[256, 128],
            mlp_dropout=0.3
        )

        # Load LoRA weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(self.device)
        model.eval()

        print("✓ Model loaded successfully")

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        )

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
            compute_metrics=self.compute_metrics
        )

        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)

        test_results = trainer.evaluate(test_dataset)

        # Print metrics
        print("\n📊 TEST METRICS:")
        print("-" * 70)
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                print(f"{key:.<50} {value:.4f}")

        # Log to W&B
        if wandb.run is not None:
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

        # Log confusion matrix to W&B
        if wandb.run is not None:
            from sklearn.metrics import confusion_matrix
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


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT with LoRA + MLP for EDU segmentation"
    )

    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--mlp_dims", nargs="+", type=int, default=[256, 128],
                        help="MLP hidden layer dimensions")
    parser.add_argument("--mlp_dropout", type=float, default=0.3,
                        help="Dropout rate in MLP")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./output-2/edu_segmenter")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # W&B configuration
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="edu-segmentation",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="W&B run name (auto-generated if empty)")

    return parser.parse_args()


def main():
    """Complete training pipeline with W&B logging"""

    print("\n" + "=" * 35)
    print("EDU SEGMENTATION: BERT + LoRA + MLP")
    print("=" * 35)

    args = parse_args()

    # Extract configuration
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    # Initialize W&B
    if args.use_wandb:
        run_name = args.wandb_run_name or f"{MODEL_NAME.replace('/', '-')}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "mlp_hidden_dims": args.mlp_dims,
                "mlp_dropout": args.mlp_dropout,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "early_stopping_patience": args.early_stopping_patience,
            }
        )
        print(f"✓ W&B initialized: {args.wandb_project}/{run_name}")

    # Step 1: Download dataset
    print("\n" + "=" * 70)
    print("STEP 1: Download Dataset")
    print("=" * 70)
    download_success = download_dataset()

    if not download_success:
        print("\n❌ Dataset download failed!")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        return

    # Step 2: Load datasets
    print("\n" + "=" * 70)
    print("STEP 2: Load Datasets")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, dev_dataset, test_dataset, _ = load_datasets(tokenizer)

    # Log dataset info to W&B
    if args.use_wandb:
        wandb.config.update({
            "train_size": len(train_dataset),
            "dev_size": len(dev_dataset),
            "test_size": len(test_dataset)
        })

    # Step 3: Initialize model
    print("\n" + "=" * 70)
    print("STEP 3: Initialize Model")
    print("=" * 70)
    device = 'cuda' if torch.cuda.is_available() else \
        'mps' if torch.backends.mps.is_available() else 'cpu'

    bert_model = BERTFineTuning(
        model_name=MODEL_NAME,
        num_labels=2,
        device=device,
        mlp_hidden_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
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
        learning_rate=LEARNING_RATE,
        save_every_n_epochs=args.save_every_n_epochs,
        early_stopping_patience=args.early_stopping_patience
    )

    # Step 5: Test evaluation
    print("\n" + "=" * 70)
    print("STEP 5: Final Test Set Evaluation")
    print("=" * 70)

    best_model_path = Path(OUTPUT_DIR) / "best_model"
    test_results = bert_model.evaluate_test_set(
        test_dataset=test_dataset,
        model_path=str(best_model_path)
    )

    # Final summary
    print("\n" + "=*=" * 35)
    print("TRAINING PIPELINE COMPLETE!")
    print("=*=" * 35)

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
    print(f"\n✅ Model saved at: {best_model_path}")
    print(f"✅ Logs saved at: {OUTPUT_DIR}/logs")

    # Create summary table for W&B
    if args.use_wandb:
        summary_data = []
        for metric in metrics:
            metric_name = metric.replace('eval_', '').capitalize()
            summary_data.append([
                metric_name,
                eval_results.get(metric, 0),
                test_results.get(metric, 0)
            ])

        wandb.log({
            "final_results_table": wandb.Table(
                columns=["Metric", "Validation", "Test"],
                data=summary_data
            )
        })

        wandb.finish()
        print("✅ W&B run completed")


if __name__ == "__main__":
    main()