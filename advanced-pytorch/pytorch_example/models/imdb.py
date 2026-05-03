"""IMDB review model factory."""

from flwr.common.typing import UserConfig

from pytorch_example.models._common import TextClassifier, get_int


def create_model(config: UserConfig | None = None):
    """Create a sentiment model for IMDB reviews."""
    return TextClassifier(
        vocab_size=get_int(config, "vocab-size", 20000),
        num_classes=get_int(config, "num-classes", 2),
        embedding_dim=get_int(config, "embedding-dim", 128),
        hidden_dim=get_int(config, "hidden-dim", 128),
    )
