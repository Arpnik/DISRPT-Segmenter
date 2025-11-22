import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from peft import get_peft_model, PeftModel, AutoPeftModelForTokenClassification
import wandb
from transformers.modeling_outputs import TokenClassifierOutput

from com.disrpt.segmenter.dataset_prep import download_dataset, load_datasets
import warnings

from com.disrpt.segmenter.utils.Helper import compute_metrics, evaluate_test_set, _get_device, compute_class_weights, \
    WeightedLossTrainer
from com.disrpt.segmenter.utils.lora_config import LoRAConfigBuilder
from com.disrpt.segmenter.utils.wandb_config import WandbEpochMetricsCallback

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

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


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
            lora_config_builder: LoRAConfigBuilder = None,
            class_1_weight_multiplier=0.7,
            use_wandb=False):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of output classes (2 for EDU segmentation)
            device: Device for training
            mlp_hidden_dims: Hidden layer dimensions for MLP
            mlp_dropout: Dropout rate in MLP
        """
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_wandb = use_wandb
        self.class_1_weight_multiplier = class_1_weight_multiplier
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout


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

        # Determine LoRA target modules
        lora_config = lora_config_builder.build(model)

        # Apply LoRA
        self.model = get_peft_model(model, lora_config)

        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()

        # Move to device
        self.model = self.model.to(device)

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
            early_stopping_threshold=0.001,
            resume_from_checkpoint=False  # ADDED: Resume parameter
    ):
        """
        Train model with LoRA + MLP classifier.
        Logs all metrics to W&B.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ADDED: Check for existing checkpoints
        checkpoint_dir = None
        if resume_from_checkpoint:
            checkpoints = list(output_path.glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                checkpoint_dir = str(latest_checkpoint)
                print(f"\n🔄 Resuming from checkpoint: {checkpoint_dir}")
            else:
                print("\n🆕 No checkpoints found. Starting fresh training.")

        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model:                    {self.model_name}")
        print(f"Output directory:         {output_dir}")
        print(f"Resume from checkpoint:   {checkpoint_dir if checkpoint_dir else 'No'}")  # ADDED
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
        callbacks = [early_stopping]
        if self.use_wandb:
            callbacks.append(WandbEpochMetricsCallback())

        # Calculate weights
        class_weights = compute_class_weights(train_dataset, self.class_1_weight_multiplier)
        print(f"\n⚖️ Class Weights Applied:")
        print(f"   Label 0 (Continue): {class_weights[0]:.4f}")
        print(f"   Label 1 (Start):    {class_weights[1]:.4f}")
        print(f"   Ratio (1/0):        {class_weights[1] / class_weights[0]:.4f}x")

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        print("\n" + "🚀" * 35)
        print("TRAINING STARTED")
        print("🚀" * 35 + "\n")

        # MODIFIED: Pass checkpoint_dir to trainer.train()
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)

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

        model_config = {
            "model_type": "mlp_classifier",  # Identifier for MLP model
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "mlp_dropout": self.mlp_dropout
        }
        with open(final_model_dir / "chunker_config.json", 'w') as f:
            json.dump(model_config, f, indent=2)

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
    parser.add_argument("--lora_layers", type=str, default="auto",
        help="Comma-separated LoRA target modules (e.g. 'query,value') or 'auto' to detect automatically"
    )

    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./output-2/edu_segmenter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_every_n_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    # ADDED: Checkpoint resume argument
    parser.add_argument("--resume_from_checkpoint", action="store_true",
                        help="Resume training from last checkpoint if available")

    # W&B configuration
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="edu-segmentation",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="W&B run name (auto-generated if empty)")
    parser.add_argument("--wandb_group", type=str, default="",
                        help="W&B run group")

    parser.add_argument("--class_1_weight_multiplier", type=float, default=0.7,
                        help="Multiply class 1 weight (>1 improves recall, try 0.5-1.5)")

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
            group=args.wandb_group if args.wandb_group else None,
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
    device = _get_device()
    print("Device for training:", device)

    # Create LoRA configuration builder
    lora_config_builder = LoRAConfigBuilder.from_string(
        model_name=MODEL_NAME,
        target_string=args.lora_layers,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        modules_to_save=["classifier"], # Train MLP alongside LoRA
        bias="all"
    )

    bert_model = BERTFineTuning(
        model_name=MODEL_NAME,
        num_labels=2,
        device=device,
        mlp_hidden_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        use_wandb=args.use_wandb,
        lora_config_builder=lora_config_builder,
        class_1_weight_multiplier=args.class_1_weight_multiplier

    )

    # Step 4: Train model
    print("\n" + "=" * 70)
    print("STEP 4: Train Model")
    print("=" * 70)

    # MODIFIED: Pass resume_from_checkpoint argument
    eval_results = bert_model.train_model(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_every_n_epochs=args.save_every_n_epochs,
        early_stopping_patience=args.early_stopping_patience,
        resume_from_checkpoint=args.resume_from_checkpoint  # ADDED
    )

    # Step 5: Test evaluation
    print("\n" + "=" * 70)
    print("STEP 5: Final Test Set Evaluation")
    print("=" * 70)

    best_model_path = Path(OUTPUT_DIR) / "best_model"
    transformers.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(best_model_path)

    # Load the base model architecture
    base_model = BERTWithMLPClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        mlp_hidden_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout
    )

    # Load PEFT model (this loads LoRA + modules_to_save like classifier)
    model = PeftModel.from_pretrained(base_model, best_model_path)
    model = model.to(device)
    model.eval()
    transformers.logging.set_verbosity_warning()

    print("✓ Model loaded successfully")

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    test_results = evaluate_test_set(
        test_dataset=test_dataset,
        model_path=str(best_model_path),
        data_collator = data_collator,
        model = model,
        batch_size=BATCH_SIZE,
        use_wandb=args.use_wandb
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