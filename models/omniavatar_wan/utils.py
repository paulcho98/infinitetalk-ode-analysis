import torch

PRECISION_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expands the input tensor `x` to have the same
    number of dimensions as the `target` tensor.

    Pads `x` with singleton dimensions on the end.

    # Example

    ```
    x = torch.ones(5)
    target = torch.ones(5, 10, 30, 1, 10)

    x = expand_like(x, target)
    print(x.shape) # <- [5, 1, 1, 1, 1]
    ```

    Args:
        x (torch.Tensor): The input tensor to expand.
        target (torch.Tensor): The target tensor whose shape length
            will be matched.

    Returns:
        torch.Tensor: The expanded tensor `x` with trailing singleton
            dimensions.
    """
    x = torch.atleast_1d(x)
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x
