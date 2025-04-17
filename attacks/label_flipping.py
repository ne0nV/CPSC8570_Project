"""
label_flipping.py
Flips labels in a client's local dataset from a source class to a target class.
"""

def poison_update(subset, source_label=0, target_label=7):
    """
    Flips labels in a torch.utils.data.Subset dataset from source_label to target_label.

    description of parameters:
        subset (torch.utils.data.Subset): A Subset of a dataset with a .targets attribute
        source_label (int): Class label to flip from (airplane)
        target_label (int): Class label to flip to (horse)
    """
    dataset = subset.dataset
    targets = dataset.targets

    # make sure  targets is a mutable list
    if not isinstance(targets, list):
        targets = list(targets)

    for idx in subset.indices:
        if targets[idx] == source_label:
            targets[idx] = target_label

    dataset.targets = targets

