# import torch
# from torchvision import models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch import nn, optim
#
# # Check if GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Define image loading and preprocessing
# def load_image(img_path, max_size=400, shape=None):
#     image = Image.open(img_path).convert('RGB')
#
#     if max(image.size) > max_size:
#         size = max_size
#     else:
#         size = max(image.size)
#
#     if shape is not None:
#         size = shape
#
#     in_transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))
#     ])
#
#     image = in_transform(image)[:3, :, :].unsqueeze(0)
#
#     return image
#
#
# # Define model and feature extraction
# class VGGFeatures(nn.Module):
#     def __init__(self):
#         super(VGGFeatures, self).__init__()
#         self.chosen_features = ['0', '5', '10', '19', '28']
#         self.model = models.vgg19(pretrained=True).features[:29]
#
#     def forward(self, x):
#         features = []
#
#         for layer_num, layer in enumerate(self.model):
#             x = layer(x)
#             if str(layer_num) in self.chosen_features:
#                 features.append(x)
#
#         return features
#
#
# # Load content and style images
# content = load_image('multicam.jpg').to(device)
# style = load_image('lyagushka.jpg', shape=content.shape[-2:]).to(device)
#
# # Initialize a model and set it to evaluation mode
# vgg = VGGFeatures().to(device).eval()
#
# # Get content and style features
# content_features = vgg(content)
# style_features = vgg(style)
#
# # Assume content and style weights
# content_weight = 1e3
# style_weight = 1e6
#
#
# # Define Gram Matrix
# def gram_matrix(tensor):
#     _, d, h, w = tensor.size()
#     tensor = tensor.view(d, h * w)
#     gram = torch.mm(tensor, tensor.t())
#     return gram
#
#
# # Get style Gram Matrices
# style_grams = {layer: gram_matrix(style_features[layer]) for layer in range(len(style_features))}
#
# # Create a target image and apply gradient descent
# target = content.clone().requires_grad_(True).to(device)
# optimizer = optim.Adam([target], lr=0.003)
# style_weights = {'0': 1., '5': 0.75, '10': 0.2, '19': 0.2, '28': 0.2}
#
# # Run style transfer
# for i in range(1, 3001):
#     target_features = vgg(target)
#     content_loss = torch.mean((target_features[2] - content_features[2]) ** 2)
#
#     style_loss = 0
#     # Ensure that `style_weights` keys match the indices from `chosen_features`
#     chosen_features = ['0', '5', '10', '19', '28']
#     style_weights = {layer: weight for layer, weight in zip(chosen_features, [1.0, 0.75, 0.2, 0.2, 0.2])}
#
#     # Now iterate over the chosen features
#     for layer in chosen_features:
#         layer_idx = int(layer)
#         target_feature = target_features[chosen_features.index(layer)]
#         target_gram = gram_matrix(target_feature)
#         style_gram = style_grams[chosen_features.index(layer)]
#         _, d, h, w = target_feature.size()  # Get the dimensions of the current target feature map
#         layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
#         style_loss += layer_style_loss / (d * h * w)
#
#     total_loss = content_weight * content_loss + style_weight * style_loss
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()
#
#     if i % 500 == 0:
#         print('Iteration {}, Total loss: {}'.format(i, total_loss.item()))
#
# # Display the target image
# plt.imshow(target.to('cpu').clone().detach().numpy().squeeze())
# plt.show()
#
# print('Iteration {}, Total loss: {}'.format(i, total_loss.item()))
#
# # Display the target image
# plt.imshow(target.to('cpu').clone().detach().numpy().squeeze())
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import vgg19

style_image_path = 'lyagushka.jpg'
base_image_path = 'flattened_luminance_image.jpg'


a = plt.imread(base_image_path)
b = plt.imread(style_image_path)


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def coste_estilo(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def coste_contenido(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

from keras.utils import plot_model

model = vgg19.VGG19(weights="imagenet", include_top=False)

from keras import Model

outputs_dict= dict([(layer.name, layer.output) for layer in model.layers])

feature_extractor = Model(inputs=model.inputs, outputs=outputs_dict)

capas_estilo = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

capas_contenido = "block5_conv2"

content_weight = 2.5e-8
style_weight = 1e-6

def loss_function(combination_image, base_image, style_reference_image):

    # 1. Combine all the images in the same tensioner.
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )

    # 2. Get the values in all the layers for the three images.
    features = feature_extractor(input_tensor)

    #3. Inicializar the loss

    loss = tf.zeros(shape=())

    # 4. Extract the content layers + content loss
    layer_features = features[capas_contenido]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = loss + content_weight * coste_contenido(
        base_image_features, combination_features
    )
    # 5. Extraer the style layers + style loss
    for layer_name in capas_estilo:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = coste_estilo(style_reference_features, combination_features)
        loss += (style_weight / len(capas_estilo)) * sl

    return loss


import tensorflow as tf

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = loss_function(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads



import keras
import numpy as np

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):

    # Convertimos el tensor en Array
    x = x.reshape((img_nrows, img_ncols, 3))

    # Hacemos que no tengan promedio 0
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Convertimos de BGR a RGB.
    x = x[:, :, ::-1]

    # Nos aseguramos que están entre 0 y 255
    x = np.clip(x, 0, 255).astype("uint8")

    return x


from datetime import datetime

def result_saver(iteration):
  # Create name
  now = datetime.now()
  now = now.strftime("%Y%m%d_%H%M%S")
  #model_name = str(i) + '_' + str(now)+"_model_" + '.h5'
  image_name = 'output30042024/'+ str(i) + '_' + str(now)+"_image" + '.png'

  # Save image
  img = deprocess_image(combination_image.numpy())
  keras.preprocessing.image.save_img(image_name, img)


from keras.optimizers import SGD

width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

optimizer = SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000

for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 10 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        result_saver(i)