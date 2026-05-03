"""UCI Adult tabular model factory."""

from flwr.common.typing import UserConfig

from pytorch_example.models._common import TabularMLP, get_int


def create_model(config: UserConfig | None = None):
    """Create a tabular classifier for UCI Adult."""
    return TabularMLP(
        input_dim=get_int(config, "input-dim", 108),
        num_classes=get_int(config, "num-classes", 2),
        hidden_dim=get_int(config, "hidden-dim", 128),
    )
