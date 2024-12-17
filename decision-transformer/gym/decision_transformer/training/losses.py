import torch

def sequence_loss(predictions, targets):
    """
    Пример функции потерь: среднеквадратическая ошибка.
    """
    return torch.nn.functional.mse_loss(predictions, targets)
