#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# #Bilateral Method to reduce image noise

# In[14]:


# Read the image.
img = cv2.imread('nature.jpeg')

# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)

# Save the output.
cv2.imwrite('nature_qqq.jpeg', bilateral)

# # Convert RGB into CIE Lab color space

# In[42]:


from PIL import Image
import numpy as np
import cv2

# Open the image
img = Image.open('nature.jpeg').convert('RGB')

# Convert the image to a NumPy array
img_array = np.array(img)

# Convert RGB to BGR (OpenCV uses BGR)
img_array = img_array[:, :, ::-1]

# Convert from BGR to CIELAB
lab_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2Lab)

# Now, lab_image contains the CIELAB values of the image
# cv2.imwrite('new_flower2.jpg', lab_image)
pixels = lab_image.reshape((-1, 3))  # Calculate the number of rows for me, 3 cols
# Seprate group
num_group = 10

# Based on the num_group value seprate into number of groups
kmeans = KMeans(n_clusters=num_group)
kmeans.fit(pixels)

# Get the colors (in Lab color space)
colors_lab = kmeans.cluster_centers_
colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_Lab2RGB)[0][0] for color in colors_lab]

# Display the colors as a palette
plt.figure(figsize=(8, 6))
palette = np.zeros((50, 50 * num_group, 3), dtype=np.uint8)

for i, color in enumerate(colors_rgb):
    palette[:, i * 50:(i + 1) * 50, :] = color

plt.imshow(palette)
plt.axis('off')
plt.show()

# In[40]:


from skimage import io, segmentation, color, graph
import skimage.future
import numpy as np


# Define your functions _weight_mean_color and merge_mean_color here
def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


# Load your own image instead of using data.coffee()
img = io.imread('nature.jpeg')  # Make sure the path to your image is correct

# Apply SLIC segmentation to your image
labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)

# Create a RAG using the mean color
g = graph.rag_mean_color(img, labels, mode='distance')  # Ensure mode is correctly set

# Merge similar regions based on color
labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

# Create an output image showing the merged regions
out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

# Display the output image
io.imshow(out)
io.show()
io.imsave('nature_q.jpeg', out)

# # RAG Comparison (Not complete)

# In[31]:


from skimage import data, color
from skimage.segmentation import felzenszwalb, slic, quickshift
import skimage.future
import numpy as np
from matplotlib import pyplot as plt


def show_img(img, algo, side_by_side=False):
    width = 10.0
    height = img[0].shape[0] * width / img[0].shape[1]
    if side_by_side:
        pad = 5.0
        f = plt.figure(figsize=(width * 2 + pad, height))
        f.add_subplot(1, 2, 1)
        plt.imshow(img[0])  # plt.imshow(np.rot90(imgRr,2))
        plt.title(algo[0])
        f.add_subplot(1, 2, 2)
        plt.imshow(img[1])
        plt.title(algo[1])
    else:
        # just display one RGB image
        f = plt.figure(figsize=(width, height))
        plt.imshow(img)
        plt.title(algo)
    plt.show(block=True)


img = cv2.imread('nature.jpeg')

labels = [quickshift(img, kernel_size=3, max_dist=6, ratio=0.5), slic(img, compactness=30, n_segments=400),
          felzenszwalb(img, scale=100, sigma=0.5, min_size=50)]
label_rgbs = [color.label2rgb(label, img, kind='avg') for label in labels]
algos = [["Quickshift", "SLIC (K-Means)", "Felzenszwalb"],
         [["Quickshift Before RAG", "Quickshift After RAG"], ["SLIC (K-Means) Before RAG", "SLIC (K-Means) After RAG"],
          ["Felzenszwalb Before RAG", "Felzenszwalb After RAG"]]]

rags = [graph.rag_mean_color(img, label) for label in labels]
edges_drawn_all = [plt.colorbar(graph.show_rag(label, rag, img)).set_label(algo) for label, rag, algo in
                   zip(labels, rags, algos[0])]

for edge_drawn in edges_drawn_all:
    plt.show()

# only display edges with weight > thresh
final_labels = [graph.cut_threshold(label, rag, 29) for label, rag in zip(labels, rags)]
final_label_rgbs = [color.label2rgb(final_label, img, kind='avg') for final_label in final_labels]

for label_rgb, final_label_rgb, algo in zip(label_rgbs, final_label_rgbs, algos[1]):
    show_img((label_rgb, final_label_rgb), algo, side_by_side=True)

# In[ ]:




