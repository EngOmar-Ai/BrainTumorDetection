from torch import Tensor
import torch

from train import model

def invoke(tensor: Tensor):
    """
    Perform inference on a batch of preprocessed brain MRI images.

    The input tensor is validated to ensure that it follows the expected
    ResNet-50 input format of (batch_size, 3, 224, 224). The model is then
    switched to evaluation mode and inference is performed without tracking
    gradients.

    The model's output logits are converted into class probabilities using
    softmax. The class with the highest probability is selected as the
    prediction, while its probability is returned as the prediction
    confidence.

    The function returns the predicted tumor category, the confidence of the
    prediction as a percentage, and the probability assigned to each of the
    four supported classes.

    Args:
        tensor (Tensor): A batch of preprocessed MRI images with shape
            (batch_size, 3, 224, 224).

    Returns:
        dict: A dictionary containing:
            - ``class``: The predicted brain tumor category.
            - ``confidence``: The confidence of the predicted class as a
              percentage.
            - ``probabilities``: A dictionary mapping each class name to its
              predicted probability.

    Raises:
        ValueError: If the input tensor does not have the expected channel
            count or spatial dimensions.

    Note:
        The input tensor should already have undergone the same preprocessing
        and ImageNet normalization used during model training.
    """

    if tensor.shape[1] != 3 or tensor.shape[2] != 224 or tensor.shape[3] != 224:
        raise ValueError('Image Tensor Should Be In The Shape Batch, 3, 224, 224')

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(logits, dim=1)

        confidence, index = torch.max(probabilities, dim=1)
        confidence, index = confidence.item(), index.item()

    classes = ['glioma', 'meningioma', 'healthy', 'pituitary']
    probability = probabilities[0].tolist()

    return {'class': classes[index], 'confidence': confidence * 100, 'probabilities': {classes[i]: round(probability[i],5) for i in range(len(classes))}}

if __name__ == "__main__":
    ...