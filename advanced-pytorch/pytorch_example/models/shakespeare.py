"""Shakespeare text model factory."""

from flwr.common.typing import UserConfig

from pytorch_example.models._common import TextClassifier, get_int


def create_model(config: UserConfig | None = None):
    """Create a text classifier for Shakespeare samples."""
    return TextClassifier(
        vocab_size=get_int(config, "vocab-size", 128),
        num_classes=get_int(config, "num-classes", 80),
        embedding_dim=get_int(config, "embedding-dim", 128),
        hidden_dim=get_int(config, "hidden-dim", 128),
    )
