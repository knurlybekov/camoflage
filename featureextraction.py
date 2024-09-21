# import cv2
# import torch
# from torchvision.models import vgg19
# import torchvision.transforms as transforms
#
#
# def load_model_and_extract_features(image, device):
#     """
#     Load VGG19 model and extract features from the specified layers.
#
#     Args:
#     image (PIL.Image): The image for which to extract the features.
#     device (torch.device): The device to run the model on.
#
#     Returns:
#     dict: A dictionary with layer names as keys and extracted features as values.
#     """
#     # Define the layers to extract features from
#     layers = ['conv1_1', 'conv3_1', 'conv5_1', 'conv7_1', 'conv9_1']
#
#     # Load the VGG19 model pre-trained on ImageNet data
#     vgg = vgg19(pretrained=True).features
#
#     # Replace MaxPool layers with AvgPool
#     for i, layer in enumerate(vgg):
#         if isinstance(layer, torch.nn.MaxPool2d):
#             vgg[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#
#     # Move the model to the specified device
#     vgg = vgg.to(device).eval()
#
#     # Process the image for VGG19
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = preprocess(image).unsqueeze(0).to(device)
#
#     # Extract features
#     features = {}
#     x = image
#     for name, layer in vgg._modules.items():
#         x = layer(x)
#         if name in layers:
#             features[name] = x
#
#     return features
#
# # Example usage:
# image = cv2.imread('pines.jpg')
# device = torch.device("cpu")
# features = load_model_and_extract_features(image, device)
# # Note: Replace 'your_image' with your actual PIL image object

import torch
from PIL import ImageFilter, Image
from matplotlib import pyplot as plt

# from PIL import Image
# from matplotlib import pyplot as plt
# from torchvision import models, transforms
# from torchvision.models import VGG19_Weights
#
# def load_vgg19_model():
#     """
#     Load the VGG19 model pre-trained on ImageNet.
#     """
#     # Load VGG19 pre-trained model
#     model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
#
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False
#
#     return model
#
# def get_features(image, model, layers=None):
#     """
#     Extract features from an image using the specified layers of the VGG19 model.
#     """
#     if layers is None:
#         layers = {'0': 'conv1_1', '5': 'conv3_1', '10': 'conv5_1', '19': 'conv7_1', '28': 'conv9_1'}
#
#     features = {}
#     x = image
#     # Model children are the sequential layers of VGG19
#     for name, layer in model._modules.items():
#         x = layer(x)
#         if name in layers:
#             features[layers[name]] = x
#
#     return features
#
# # Load the VGG19 model
# vgg = load_vgg19_model()
#
# def preprocess_image(image_path, target_size=(224, 224)):
#     """
#     Preprocess the image to be suitable for feature extraction with VGG19.
#
#     Args:
#     image_path (str): Path to the image file.
#     target_size (tuple): Desired size for the image.
#
#     Returns:
#     torch.Tensor: Preprocessed image tensor.
#     """
#     # Load the image
#     image = Image.open(image_path)
#
#     # Convert to RGB if the image has an alpha channel
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#
#     # Define the transformation
#     transform = transforms.Compose([
#         transforms.Resize(target_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # Apply the transformation
#     image = transform(image).unsqueeze(0)
#
#     return image
#
#
# def visualize_feature_map(feature_map, filter_index=0):
#     """
#     Visualize a single filter from a feature map.
#
#     Args:
#     feature_map (torch.Tensor): The feature map tensor.
#     filter_index (int): Index of the filter to visualize.
#     """
#     # Ensure the feature map is detached and moved to CPU
#     feature_map = feature_map.detach().cpu()
#
#     # Select the specified filter
#     if feature_map.shape[1] > filter_index:
#         filter_activation = feature_map[0, filter_index, :, :]
#     else:
#         raise ValueError("Filter index is out of range.")
#
#     # Convert to numpy array and normalize
#     filter_activation = filter_activation.numpy()
#     filter_activation = (filter_activation - filter_activation.min()) / \
#                         (filter_activation.max() - filter_activation.min())
#
#
#     # Display the feature map
#     plt.imshow(filter_activation, cmap='viridis')
#     plt.colorbar()
#     plt.show()
#
# # Example usage
# content_image_path = 'multicam.jpg'
# style_image_path = 'pines.jpg'
#
# content_image = preprocess_image(content_image_path)
# style_image = preprocess_image(style_image_path)
# # Example: Extract features from the content and style images
# # Assuming content_image and style_image are already preprocessed tensors
# content_features = get_features(content_image, vgg)
# style_features = get_features(style_image, vgg)
#
# visualize_feature_map(content_features['conv1_1'], filter_index=0)


# print("Content Features:")
# for layer, feature in content_features.items():
#     print(f"{layer}: {feature.shape}")
#
# # Printing style features
# print("\nStyle Features:")
# for layer, feature in style_features.items():
#     print(f"{layer}: {feature.shape}")
