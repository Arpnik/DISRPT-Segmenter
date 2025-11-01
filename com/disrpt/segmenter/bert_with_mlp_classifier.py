# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoConfig
#
#
# class BERTWithMLPClassifier(nn.Module):
#     """
#     BERT/DistilBERT with a Multi-Layer Perceptron (MLP) classifier head
#     for EDU boundary detection.
#
#     Architecture:
#         BERT Encoder → Dropout → MLP (hidden layers) → Output (2 classes)
#     """
#
#     def __init__(
#             self,
#             model_name,
#             num_labels=2,
#             hidden_dims=[256, 128],  # MLP hidden layer sizes
#             dropout_rate=0.3,
#             activation='relu'
#     ):
#         """
#         Args:
#             model_name: HuggingFace model name (e.g., 'distilbert-base-uncased')
#             num_labels: Number of output classes (2 for binary EDU segmentation)
#             hidden_dims: List of hidden layer dimensions for MLP
#             dropout_rate: Dropout probability
#             activation: Activation function ('relu', 'gelu', 'tanh')
#         """
#         super(BERTWithMLPClassifier, self).__init__()
#
#         # Load BERT encoder (without classification head)
#         config = AutoConfig.from_pretrained(model_name)
#         self.bert = AutoModel.from_pretrained(model_name, config=config)
#         self.hidden_size = config.hidden_size  # 768 for BERT-base/DistilBERT
#
#         # Select activation function
#         activations = {
#             'relu': nn.ReLU(),
#             'gelu': nn.GELU(),
#             'tanh': nn.Tanh()
#         }
#         self.activation = activations.get(activation.lower(), nn.ReLU())
#
#         # Build MLP classifier
#         layers = []
#         input_dim = self.hidden_size
#
#         # Hidden layers
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(input_dim, hidden_dim),
#                 self.activation,
#                 nn.Dropout(dropout_rate)
#             ])
#             input_dim = hidden_dim
#
#         # Output layer
#         layers.append(nn.Linear(input_dim, num_labels))
#
#         self.classifier = nn.Sequential(*layers)
#
#         # Store config
#         self.num_labels = num_labels
#         self.config = config
#
#         print(f"\n🏗️  MLP Classifier Architecture:")
#         print(f"   Input:  {self.hidden_size} (BERT hidden size)")
#         for i, dim in enumerate(hidden_dims):
#             print(f"   Hidden Layer {i + 1}: {dim} → {activation.upper()} → Dropout({dropout_rate})")
#         print(f"   Output: {num_labels} (classification logits)")
#
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         """
#         Forward pass with optional loss calculation.
#
#         Args:
#             input_ids: Token IDs (batch_size, seq_length)
#             attention_mask: Attention mask (batch_size, seq_length)
#             labels: Ground truth labels (batch_size, seq_length)
#
#         Returns:
#             Dictionary with 'loss' (if labels provided) and 'logits'
#         """
#         # Get BERT encodings
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#
#         # Get token-level hidden states
#         # Shape: (batch_size, seq_length, hidden_size)
#         sequence_output = outputs.last_hidden_state
#
#         # Pass through MLP classifier
#         # Shape: (batch_size, seq_length, num_labels)
#         logits = self.classifier(sequence_output)
#
#         loss = None
#         if labels is not None:
#             # Cross-entropy loss (ignores -100 labels)
#             loss_fct = nn.CrossEntropyLoss()
#
#             # Flatten for loss calculation
#             # Only compute loss on non-ignored tokens
#             active_loss = attention_mask.view(-1) == 1
#             active_logits = logits.view(-1, self.num_labels)[active_loss]
#             active_labels = labels.view(-1)[active_loss]
#
#             # Filter out -100 labels
#             valid_indices = active_labels != -100
#             if valid_indices.sum() > 0:
#                 loss = loss_fct(
#                     active_logits[valid_indices],
#                     active_labels[valid_indices]
#                 )
#             else:
#                 loss = torch.tensor(0.0, device=logits.device)
#
#         return {
#             'loss': loss,
#             'logits': logits
#         }
#
