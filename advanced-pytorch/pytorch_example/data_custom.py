from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner

from pytorch_example.task import get_apply_eval_transforms, get_apply_train_transforms

fds = None


def load_custom_data(
    partition_id: int,
    num_partitions: int,
    config,
):
    global fds
    partitioner_id = config.get("partitioner-id", "dirichlet")
    dataset_name = config.get("dataset-name", "zalando-datasets/fashion_mnist")
    batch_size = config.get("batch-size", 32)
    feature_column = config.get("feature-column", "image")
    partition_by = config.get("partition-by", "label")
    alpha = config.get("partition-alpha", 1.0)
    seed = config.get("partition-seed", 42)
    num_shards_per_partition = config.get("num-shards-per-partition", 2)

    if fds is None:
        if partitioner_id == "iid":
            partitioner = IidPartitioner(seed=seed, num_partitions=num_partitions)
        elif partitioner_id == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=partition_by,
                alpha=alpha,
                seed=seed,
            )
        elif partitioner_id == "shard":
            partitioner = ShardPartitioner(
                num_partitions=num_partitions, 
                partition_by=partition_by,
                num_shards_per_partition=num_shards_per_partition,
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported partitioner_id: {partitioner_id}")
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    train_partition = partition_train_test["train"].with_transform(
        get_apply_train_transforms(feature_column)
    )
    test_partition = partition_train_test["test"].with_transform(
        get_apply_eval_transforms(feature_column)
    )

    trainloader = DataLoader(
        train_partition,
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        test_partition,
        batch_size=batch_size,
    )

    return trainloader, testloader
