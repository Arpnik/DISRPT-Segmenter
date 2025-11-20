from typing import List, Optional
import torch.nn as nn
from peft import LoraConfig, TaskType


class LoRAConfigBuilder:
    """
    Builder class for creating and validating LoRA configurations for different model architectures.
    """

    # Default target modules for different model types
    DEFAULT_TARGETS = {
        'distilbert': ['q_lin', 'v_lin'],
        'bert': ['query', 'value'],
        'roberta': ['query', 'value'],
        'albert': ['query', 'value'],
        'electra': ['query', 'value'],
        'deberta': ['query_proj', 'value_proj'],
    }

    def __init__(
            self,
            model_name: str,
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Optional[List[str]] = None,
            modules_to_save: Optional[List[str]] = None,
            bias: str = "none",
            task_type: TaskType = TaskType.FEATURE_EXTRACTION
    ):
        """
        Initialize LoRA configuration builder.

        Args:
            model_name: HuggingFace model identifier
            r: LoRA rank (dimension of low-rank matrices)
            lora_alpha: LoRA scaling parameter (alpha/r gives effective learning rate)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Specific modules to apply LoRA (None = auto-detect)
            modules_to_save: Additional modules to train (e.g., classifier head)
            bias: Bias training strategy ("none", "all", "lora_only")
            task_type: PEFT task type
        """
        self.model_name = model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.modules_to_save = modules_to_save or []
        self.bias = bias
        self.task_type = task_type

    def _detect_model_type(self) -> str:
        """Detect model type from model name."""
        model_lower = self.model_name.lower()

        for model_type in self.DEFAULT_TARGETS.keys():
            if model_type in model_lower:
                return model_type

        # Default to BERT-style if unknown
        return 'bert'

    def _get_default_targets(self) -> List[str]:
        """Get default target modules based on model type."""
        model_type = self._detect_model_type()
        return self.DEFAULT_TARGETS[model_type]

    def validate_targets(self, model: nn.Module) -> List[str]:
        """
        Validate that target modules exist in the model architecture.

        Args:
            model: PyTorch model to validate against

        Returns:
            List of validated target module names

        Raises:
            ValueError: If target modules don't exist in the model
        """
        # Use provided targets or auto-detect
        targets = self.target_modules or self._get_default_targets()

        # Get all module names from the model
        module_names = [name for name, _ in model.named_modules()]

        # Validate each target
        missing = []
        for target in targets:
            # Check if target appears in any module name
            if not any(target in mod for mod in module_names):
                missing.append(target)

        if missing:
            print("\n" + "=" * 70)
            print("❌ LORA TARGET VALIDATION FAILED")
            print("=" * 70)
            print(f"\nModel: {self.model_name}")
            print(f"Missing targets: {missing}")
            print(f"\n💡 Available attention-related modules:")
            print("-" * 70)

            # Show relevant modules
            relevant_keywords = ["att", "query", "key", "value", "proj", "lin", "dense"]
            relevant_modules = [
                name for name in module_names
                if any(k in name.lower() for k in relevant_keywords)
            ]

            for name in sorted(set(relevant_modules)):
                print(f"  • {name}")

            print("\n" + "=" * 70)
            raise ValueError(
                f"LoRA target modules {missing} do not exist in {self.model_name}. "
                f"Use modules from the list above."
            )

        print("\n✓ LoRA target validation passed")
        print(f"  Target modules: {targets}")

        return targets

    def build(self, model: nn.Module) -> LoraConfig:
        """
        Build and return validated LoRA configuration.

        Args:
            model: Model to apply LoRA to

        Returns:
            Configured LoraConfig object
        """
        # Validate targets
        validated_targets = self.validate_targets(model)

        # Create LoRA config
        config = LoraConfig(
            task_type=self.task_type,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=validated_targets,
            bias=self.bias,
            modules_to_save=self.modules_to_save
        )

        # Print configuration summary
        print("\n" + "=" * 70)
        print("LORA CONFIGURATION")
        print("=" * 70)
        print(f"  Rank (r):              {self.r}")
        print(f"  Alpha:                 {self.lora_alpha}")
        print(f"  Effective LR scale:    {self.lora_alpha / self.r:.2f}x")
        print(f"  Dropout:               {self.lora_dropout}")
        print(f"  Target modules:        {', '.join(validated_targets)}")
        print(f"  Modules to save:       {', '.join(self.modules_to_save) if self.modules_to_save else 'None'}")
        print(f"  Bias:                  {self.bias}")
        print(f"  Task type:             {self.task_type}")
        print("=" * 70)

        return config

    @classmethod
    def from_string(
            cls,
            model_name: str,
            target_string: str,
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            modules_to_save: Optional[List[str]] = None,
            task_type: TaskType = TaskType.FEATURE_EXTRACTION,
            bias: str = "none"
    ) -> 'LoRAConfigBuilder':
        """
        Create LoRA config builder from comma-separated string.

        Args:
            model_name: HuggingFace model identifier
            target_string: Comma-separated target modules (e.g., "query,value")
                          or "auto" to auto-detect
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            modules_to_save: Additional modules to train

        Returns:
            LoRAConfigBuilder instance
        """
        if target_string.lower() == "auto":
            target_modules = None
        else:
            target_modules = [x.strip() for x in target_string.split(",")]

        return cls(
            model_name=model_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type=task_type,
            bias=bias
        )