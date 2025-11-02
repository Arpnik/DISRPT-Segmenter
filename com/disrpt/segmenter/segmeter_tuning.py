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
from com.disrpt.segmenter.dataset_prep import download_dataset, load_datasets
import warnings
warnings.filterwarnings("ignore")


class BERTFineTuning:
    """
    Enhanced BERT fine-tuning with configurable checkpointing,
    early stopping, and comprehensive evaluation.
    """

    def __init__(self, model_name, num_labels=2, device='cuda'):
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels

        print("\n" + "=" * 70)
        print(f"Initializing {model_name} with LoRA")
        print("=" * 70)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"] if "distilbert" in model_name.lower()
            else ["query", "value"],
            bias="none"
        )

        self.model = get_peft_model(model, lora_config)
        print("\n📊 Trainable Parameters:")
        self.model.print_trainable_parameters()

    @staticmethod
    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            for pred_label, true_label in zip(prediction, label):
                if true_label != -100:
                    true_predictions.append(pred_label)
                    true_labels.append(true_label)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            average='binary',
            pos_label=1,
            zero_division=0
        )
        acc = accuracy_score(true_labels, true_predictions)

        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # Log metrics to wandb if active
        if wandb.run is not None:
            wandb.log(metrics)

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

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,

            # Logging + evaluation + wandb integration
            logging_dir=f'{output_dir}/logs',
            logging_steps=20,
            logging_strategy="steps",
            eval_strategy="epoch",
            report_to="wandb" if wandb.run is not None else "none",

            # Checkpoints
            save_strategy="epoch",
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )

        print("\n🚀 TRAINING STARTED 🚀\n")
        train_result = trainer.train()
        print("\n✅ TRAINING COMPLETED ✅")

        # Log training metrics to W&B
        if wandb.run is not None:
            wandb.log(train_result.metrics)

        print("\nFINAL TRAINING METRICS")
        print("=" * 70)
        for key, value in train_result.metrics.items():
            print(f"{key:.<50} {value:.4f}")

        print("\nVALIDATION METRICS")
        print("=" * 70)
        eval_results = trainer.evaluate()
        if wandb.run is not None:
            wandb.log(eval_results)
        for key, value in eval_results.items():
            print(f"{key:.<50} {value:.4f}")

        final_model_dir = output_path / "best_model"
        print(f"\nSaving best model to: {final_model_dir}")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print("✓ Model saved successfully")

        return eval_results

    def evaluate_test_set(self, test_dataset, model_path, batch_size=32):
        print("\nLOADING MODEL FOR TEST EVALUATION")
        print("=" * 70)
        print(f"Model path: {model_path}")
        print(f"Test examples: {len(test_dataset)}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=self.num_labels
        ).to(self.device)

        print("✓ Model loaded successfully")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

        eval_args = TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),
            report_to="wandb" if wandb.run is not None else "none"
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        print("\nTEST SET EVALUATION")
        print("=" * 70)
        test_results = trainer.evaluate(test_dataset)

        if wandb.run is not None:
            wandb.log({"test_" + k: v for k, v in test_results.items()})

        for key, value in test_results.items():
            print(f"{key:.<50} {value:.4f}")

        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=2)
        true_labels = predictions.label_ids

        flat_preds, flat_labels = [], []
        for preds, labels in zip(pred_labels, true_labels):
            for pred, label in zip(preds, labels):
                if label != -100:
                    flat_preds.append(pred)
                    flat_labels.append(label)

        print("\nDETAILED CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(
            flat_labels,
            flat_preds,
            target_names=['EDU Continue (0)', 'EDU Start (1)'],
            digits=4
        ))

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
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="edu-segmentation")
    parser.add_argument("--wandb_run_name", type=str, default="")
    return parser.parse_args()


def main():
    print("\nEDU SEGMENTATION MODEL TRAINING")
    print("=" * 35)
    args = parse_args()

    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    LEARNING_RATE = args.learning_rate

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{MODEL_NAME.replace('/', '-')}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}",
            config={
                "model_name": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            }
        )

    print("\nSTEP 1: Download Dataset")
    download_success = download_dataset()
    if not download_success:
        print("\n❌ Dataset download failed!")
        return

    print("\nSTEP 2: Load Datasets")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, dev_dataset, test_dataset, distribution = load_datasets(tokenizer)

    print("\nSTEP 3: Initialize Model")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    bert_model = BERTFineTuning(MODEL_NAME, num_labels=2, device=device)

    print("\nSTEP 4: Train Model")
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

    print("\nSTEP 5: Evaluate on Test Set")
    best_model_path = Path(OUTPUT_DIR) / "best_model"
    test_results = bert_model.evaluate_test_set(test_dataset=test_dataset, model_path=str(best_model_path))

    if args.use_wandb:
        wandb.log({"final_eval_metrics": eval_results, "final_test_metrics": test_results})
        wandb.finish()


if __name__ == "__main__":
    main()
