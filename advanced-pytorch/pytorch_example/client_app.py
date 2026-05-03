"""pytorch-example: A Flower / PyTorch app."""

from importlib import import_module

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorch_example.task import Net
from pytorch_example.task import test as test_fn
from pytorch_example.task import train as train_fn

# Flower ClientApp
app = ClientApp()
classification_head_name = "classification-head"


def save_layer_weights_to_state(state: RecordDict, net: Net):
    """Save last layer weights to state."""
    state[classification_head_name] = ArrayRecord(net.fc2.state_dict())


def load_layer_weights_from_state(state: RecordDict, net: Net):
    """Load last layer weights from state and applies them to the model."""
    if classification_head_name not in state:
        return

    # Restore this client's saved classification head
    state_dict = state[classification_head_name].to_torch_state_dict()
    net.fc2.load_state_dict(state_dict, strict=True)


def get_data_loaders(context: Context):
    """Load client data using the configured data-loader callback."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    module_name, function_name = context.run_config["load-data-fn"].split(":")
    load_data_fn = getattr(import_module(module_name), function_name)
    return load_data_fn(partition_id, num_partitions, context.run_config)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load model and apply received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    trainloader, _ = get_data_loaders(context)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Save classification head in `context.state` to use in future rounds
    save_layer_weights_to_state(context.state, model)

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
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # Restore this client's previously saved classification layer weights
    # (no action if this is the first round it participates in)
    load_layer_weights_from_state(context.state, model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    _, valloader = get_data_loaders(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
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
