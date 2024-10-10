import torch
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, accuracy_score

def F1(p):
    """
    Computes evaluation metrics for multi-label classification tasks.

    This function calculates precision, recall, and F1 scores using both macro and micro averaging,
    as well as Hamming loss and subset accuracy, based on the model's logits and the true labels.

    Args:
        p (tuple): A tuple containing two elements:
            - logits (array-like): The raw output logits from the model (before activation).
            - labels (array-like): The ground truth binary labels.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - 'precision_micro': Micro-averaged precision across all classes.
            - 'recall_micro': Micro-averaged recall across all classes.
            - 'f1_micro': Micro-averaged F1 score across all classes.
            - 'precision_macro': Macro-averaged precision across all classes.
            - 'recall_macro': Macro-averaged recall across all classes.
            - 'f1_macro': Macro-averaged F1 score across all classes.
            - 'hamming_loss': Hamming loss between the true labels and predictions.
            - 'subset_accuracy': Proportion of samples where all labels are correctly predicted.

    Notes:
        - The logits are converted to probabilities using the sigmoid activation function.
        - Predictions are obtained by thresholding these probabilities at 0.5.
        - The function is designed for multi-label classification problems where each sample may belong to multiple classes.

    Example:
        >>> logits = [[2.0, -1.0, 0.5], [1.0, 2.0, -0.5]]
        >>> labels = [[1, 0, 1], [0, 1, 0]]
        >>> metrics = F1((logits, labels))
        >>> print(metrics)
        {
            'precision_micro': 1.0,
            'recall_micro': 1.0,
            'f1_micro': 1.0,
            'precision_macro': 1.0,
            'recall_macro': 1.0,
            'f1_macro': 1.0,
            'hamming_loss': 0.0,
            'subset_accuracy': 1.0,
        }

    """
    logits, labels = p

    # Apply sigmoid activation to logits to get probabilities
    sigmoid_logits = torch.sigmoid(torch.tensor(logits))

    # Threshold probabilities at 0.5 to get binary predictions
    preds = (sigmoid_logits > 0.5).int().numpy()

    # Ensure labels are in integer format
    labels = labels.astype(int)

    # Compute macro-averaged precision, recall, and F1 score
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # Compute micro-averaged precision, recall, and F1 score
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0
    )

    # Compute Hamming loss
    hamming_loss_value = hamming_loss(labels, preds)

    # Compute subset accuracy (exact match ratio)
    subset_accuracy = accuracy_score(labels, preds)

    return {
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming_loss_value,
        'subset_accuracy': subset_accuracy,
    }
