# import cv2
# from sklearn.cluster import AgglomerativeClustering
# from PIL import Image #pillow
# import numpy as np
# from sklearn.cluster import KMeans
# import colour as cl
#
# # Read the image.
# img = cv2.imread('pine-timber-guide-0.jpg')
#
# # Apply bilateral filter with d = 15,
# # sigmaColor = sigmaSpace = 75.
# bilateral = cv2.bilateralFilter(img, 15, 75, 75)
#
# # Save the output.
# cv2.imwrite('taj_bilateral.jpg', bilateral)
#
#
#
# #image to use
# file = 'taj_bilateral.jpg'
#
# #number of clusters
# nclusters = 5
#
# #load image and get colors
# image = Image.open(file)
# pixels = np.array(list(image.getdata()))
#
# ##fit KMeans and get centroids
# kmeans = KMeans(n_clusters = nclusters)
# kmeans = kmeans.fit(pixels)
# centroids = kmeans.cluster_centers_
#
# #converts from rgb to xyz
# def rgb_to_xyz(p):
#     RGB_to_XYZ_matrix = np.array(
#         [[0.41240000, 0.35760000, 0.18050000],
#         [0.21260000, 0.71520000, 0.07220000],
#         [0.01930000, 0.11920000, 0.95050000]])
#     illuminant_RGB = np.array([0.31270, 0.32900])
#     illuminant_XYZ = np.array([0.34570, 0.35850])
#     return cl.RGB_to_XYZ(p / 255, illuminant_RGB, illuminant_XYZ,
# 							RGB_to_XYZ_matrix, 'Bradford')
#
# #converts from rgb to lab
# def rgb_to_lab(p):
#     new = rgb_to_xyz(p)
#     return cl.XYZ_to_Lab(new)
#
# #converts from xyz to rgb
# def xyz_to_rgb(p):
#     XYZ_to_RGB_matrix = np.array(
#         [[3.24062548, -1.53720797, -0.49862860],
#         [-0.96893071, 1.87575606, 0.04151752],
#         [0.05571012, -0.20402105, 1.05699594]])
#     illuminant_RGB = np.array([0.31270, 0.32900])
#     illuminant_XYZ = np.array([0.34570, 0.35850])
#     newp = cl.XYZ_to_RGB(p, illuminant_XYZ, illuminant_RGB,
# 							XYZ_to_RGB_matrix, 'Bradford')
#     return newp * 255
#
# #converts from lab to rgb
# def lab_to_rgb(p):
#     xyz = cl.Lab_to_XYZ(p)
#     return xyz_to_rgb(xyz)
#
# #kmeans
# kmeans_lab = KMeans(n_clusters = nclusters)
# kmeans_lab = kmeans_lab.fit(rgb_to_lab(pixels))
# centroids_lab = kmeans_lab.cluster_centers_
# centroids_lab = lab_to_rgb(centroids_lab)
#
# #convert to lab
# pixels_lab = rgb_to_lab(pixels)
#
# #fit clusters
# ag_clusters = AgglomerativeClustering(n_clusters=nclusters, linkage='complete')
# ag_clusters_fit = ag_clusters.fit(rgb_to_lab(pixels))
# #get centroids
# centroids_ag = []
# for i in range(nclusters):
#     center = pixels_lab[ag_clusters_fit.labels_ == i].mean(0)
#     centroids_ag.append(lab_to_rgb(center))


import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt


def load_and_preprocess_image(file_path):
    # Load image using OpenCV
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError("Image not found or invalid image file.")

    # Apply bilateral filter with d = 15, sigmaColor = sigmaSpace = 75.
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)
    return bilateral


def convert_to_cielab(image):
    # Convert image to CIELAB color space
    cielab_image = color.rgb2lab(image)
    return cielab_image


def lab_to_rgb(lab):
    # Convert LAB to RGB and denormalize values to the range [0, 255]
    rgb_normalized = color.lab2rgb(lab)
    rgb = np.clip(rgb_normalized * 255, 0, 255).astype(np.uint8)
    return rgb


def perform_clustering(image, n_clusters):
    # Convert image to LAB color space for clustering
    lab_image = convert_to_cielab(image)

    # K-Means clustering for improved efficiency
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(lab_image.reshape(-1, 3))

    # Get the cluster centroids (in LAB color space) and convert to RGB
    centroids_lab = kmeans.cluster_centers_
    centroids_rgb = lab_to_rgb(centroids_lab)

    # Get the labels for each pixel
    labels = kmeans.labels_

    return centroids_rgb, labels






# from skimage import data, io, segmentation, color
# from skimage import graph
# import numpy as np
#
#
# def _weight_mean_color(graph, src, dst, n):
#     """Callback to handle merging nodes by recomputing mean color.
#
#     The method expects that the mean color of `dst` is already computed.
#
#     Parameters
#     ----------
#     graph : RAG
#         The graph under consideration.
#     src, dst : int
#         The vertices in `graph` to be merged.
#     n : int
#         A neighbor of `src` or `dst` or both.
#
#     Returns
#     -------
#     data : dict
#         A dictionary with the `"weight"` attribute set as the absolute
#         difference of the mean color between node `dst` and `n`.
#     """
#
#     diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
#     diff = np.linalg.norm(diff)
#     return {'weight': diff}
#
#
# def merge_mean_color(graph, src, dst):
#     """Callback called before merging two nodes of a mean color distance graph.
#
#     This method computes the mean color of `dst`.
#
#     Parameters
#     ----------
#     graph : RAG
#         The graph under consideration.
#     src, dst : int
#         The vertices in `graph` to be merged.
#     """
#     graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
#     graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
#     graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
#                                       graph.nodes[dst]['pixel count'])
#
#
#
# img = cv2.imread('pines.jpg')
# labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
# g = graph.rag_mean_color(img, labels)
#
# labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
#                                    in_place_merge=True,
#                                    merge_func=merge_mean_color,
#                                    weight_func=_weight_mean_color)
#
# out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
# out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
# io.imshow(out)
# io.show()

# Path to your image  # Update with the correct path


# Preprocess the image and perform clustering
preprocessed_image = load_and_preprocess_image('kamloops.jpg')
centroids_rgb, labels = perform_clustering(preprocessed_image, n_clusters=5)

# Calculate the percentage of each color
(unique, counts) = np.unique(labels, return_counts=True)
percentages = counts / labels.size * 100

# Display the centroids and their percentages
plt.figure(figsize=(10, 3))
for i, (centroid, percentage) in enumerate(zip(centroids_rgb, percentages)):
    plt.subplot(1, 5, i+1)
    plt.imshow(centroid.reshape(1, 1, 3))
    plt.title(f'{percentage:.2f}%\nRGB: {centroid[0]}, {centroid[1]}, {centroid[2]}')
    plt.axis('off')
plt.show()

