"""CIFAR-100 model factory."""

from flwr.common.typing import UserConfig

from pytorch_example.models._common import ImageCNN, get_int


def create_model(config: UserConfig | None = None):
    """Create a model for CIFAR-100."""
    return ImageCNN(
        in_channels=get_int(config, "input-channels", 3),
        num_classes=get_int(config, "num-classes", 100),
        width=get_int(config, "cnn-width", 64),
    )
