from transformers import TrainerCallback
import wandb


class WandbEpochMetricsCallback(TrainerCallback):
    """
    Logs metrics to W&B with epoch as the x-axis for proper graph visualization.
    Handles both training and evaluation metrics.
    """

    def __init__(self):
        super().__init__()
        self.training_step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation. Logs eval/validation metrics to W&B with epoch as x-axis.
        """
        if metrics and wandb.run is not None and state.epoch is not None:
            # Create log dictionary with all eval metrics
            log_dict = {
                "epoch": state.epoch,  # This makes epoch the x-axis
                "eval/loss": metrics.get("eval_loss"),
                "eval/accuracy": metrics.get("eval_accuracy"),
                "eval/precision": metrics.get("eval_precision"),
                "eval/recall": metrics.get("eval_recall"),
                "eval/f1": metrics.get("eval_f1"),
                "eval/f1_class_0": metrics.get("eval_f1_class_0"),
                "eval/f1_class_1": metrics.get("eval_f1_class_1"),
                "eval/precision_class_0": metrics.get("eval_precision_class_0"),
                "eval/precision_class_1": metrics.get("eval_precision_class_1"),
                "eval/recall_class_0": metrics.get("eval_recall_class_0"),
                "eval/recall_class_1": metrics.get("eval_recall_class_1"),
                "eval/support_class_0": metrics.get("eval_support_class_0"),
                "eval/support_class_1": metrics.get("eval_support_class_1"),
            }

            # Remove None values (but keep epoch)
            log_dict = {k: v for k, v in log_dict.items() if v is not None}

            # Log to W&B - including epoch in the dict makes it the x-axis
            wandb.log(log_dict)

            # Print formatted output
            print(f"\n{'=' * 70}")
            print(f"📊 Epoch {state.epoch:.2f} - Evaluation Metrics")
            print(f"{'=' * 70}")

            # Print main metrics
            main_metrics = ["eval/loss", "eval/accuracy", "eval/precision", "eval/recall", "eval/f1"]
            for k in main_metrics:
                if k in log_dict:
                    print(f"  {k:<25} {log_dict[k]:.4f}")

            # Print per-class metrics
            if any("class_" in k for k in log_dict.keys()):
                print(f"\n  Per-class metrics:")
                class_metrics = [k for k in log_dict.keys() if "class_" in k and "support" not in k]
                for k in sorted(class_metrics):
                    print(f"    {k:<23} {log_dict[k]:.4f}")

            print(f"{'=' * 70}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Log training metrics with epoch as x-axis.
        Captures loss, learning rate, and gradient norm during training.
        """
        if logs and wandb.run is not None and state.epoch is not None:
            # Only process training logs (not eval logs)
            # Eval logs contain "eval_loss", training logs contain just "loss"
            if "loss" in logs and "eval_loss" not in logs:
                log_dict = {
                    "epoch": state.epoch,  # Include epoch for x-axis
                }

                # Add training metrics
                if "loss" in logs:
                    log_dict["train/loss"] = logs["loss"]
                if "learning_rate" in logs:
                    log_dict["train/learning_rate"] = logs["learning_rate"]
                if "grad_norm" in logs:
                    log_dict["train/grad_norm"] = logs["grad_norm"]

                # Log with epoch (if we have more than just epoch)
                if len(log_dict) > 1:
                    wandb.log(log_dict)

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Called at the end of each epoch.
        Can be used for additional epoch-level logging if needed.
        """
        if wandb.run is not None and state.epoch is not None:
            # Log epoch completion
            wandb.log({
                "epoch": state.epoch,
                "training/epoch_completed": state.epoch
            })
