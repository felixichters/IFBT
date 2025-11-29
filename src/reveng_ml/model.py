"""
Model definition for the RevEng-ML project.
"""
from transformers import BertConfig, BertForTokenClassification

def get_model(
    vocab_size=257,  # TODO: check this later
    # TODO: Which labels to choose?
    num_labels=3,    # for now -> 0: None, 1: Function start, 2: Function end
    hidden_size=256,
    num_attention_heads=8,
    num_hidden_layers=4,
    intermediate_size=1024
):
    """
    Initializes a new BERT model for token classification with a custom configuration.

    Args:
        vocab_size (int): Size of vocabulary 256 bytes (+ special token)
        num_labels (int): Number of labels for classification
        hidden_size (int): The dimensionality of the model's hidden layers
        num_attention_heads (int): The number of attention heads in the transformer
        num_hidden_layers (int): The number of transformer layers
        intermediate_size (int): The size of the "feed-forward" layer in the transformer

    Returns:
        A `BertForTokenClassification` model instance.
    """
    config = BertConfig(
        vocab_size=vocab_size,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        type_vocab_size=1,
    )

    model = BertForTokenClassification(config)
    return model

