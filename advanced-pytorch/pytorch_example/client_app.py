"""pytorch-example: A Flower / PyTorch app."""

from importlib import import_module

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorch_example.task import test as test_fn
from pytorch_example.task import train as train_fn

# Flower ClientApp
app = ClientApp()
classification_head_name = "classification-head"

DEFAULT_MODEL_FN = "pytorch_example.model:create_model"
DEFAULT_LOAD_DATA_FN = "pytorch_example.task:load_data"
DEFAULT_FEATURE_COLUMN = "image"
DEFAULT_TARGET_COLUMN = "label"


def import_configured_fn(path: str):
    """Import a configured callback formatted as 'module:function'."""
    module_name, function_name = path.split(":")
    return getattr(import_module(module_name), function_name)


def create_model(context: Context):
    """Create the configured model."""
    create_model_fn = import_configured_fn(
        context.run_config.get("model-fn", DEFAULT_MODEL_FN)
    )
    return create_model_fn(context.run_config)


def save_layer_weights_to_state(state: RecordDict, net, layer_name: str):
    """Save last layer weights to state."""
    if not layer_name:
        return
    layer = getattr(net, layer_name)
    state[classification_head_name] = ArrayRecord(layer.state_dict())


def load_layer_weights_from_state(state: RecordDict, net, layer_name: str):
    """Load last layer weights from state and applies them to the model."""
    if not layer_name or classification_head_name not in state:
        return

    # Restore this client's saved classification head
    state_dict = state[classification_head_name].to_torch_state_dict()
    layer = getattr(net, layer_name)
    layer.load_state_dict(state_dict, strict=True)


def get_data_loaders(context: Context):
    """Load client data using the configured data-loader callback."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    load_data_fn = import_configured_fn(
        context.run_config.get("load-data-fn", DEFAULT_LOAD_DATA_FN)
    )
    return load_data_fn(partition_id, num_partitions, context.run_config)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load model and apply received weights
    personalized_layer_name = context.run_config.get("personalized-layer-name", "fc2")
    model = create_model(context)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model, personalized_layer_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    trainloader, _ = get_data_loaders(context)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config.get("local-epochs", 1),
        msg.content["config"]["lr"],
        device,
        context.run_config.get("feature-column", DEFAULT_FEATURE_COLUMN),
        context.run_config.get("target-column", DEFAULT_TARGET_COLUMN),
    )

    # Save classification head in `context.state` to use in future rounds
    save_layer_weights_to_state(context.state, model, personalized_layer_name)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load model and apply received weights
    personalized_layer_name = context.run_config.get("personalized-layer-name", "fc2")
    model = create_model(context)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model, personalized_layer_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    _, valloader = get_data_loaders(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
        context.run_config.get("feature-column", DEFAULT_FEATURE_COLUMN),
        context.run_config.get("target-column", DEFAULT_TARGET_COLUMN),
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
