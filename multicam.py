# import random
# from PIL import Image, ImageDraw
#
# # Define your colors and percentages
# colors = [(68, 90, 79), (55, 73, 63), (51, 51, 45), (97, 115, 113), (77, 77, 70)]
# percentages = [25.36, 25.95, 18.83, 16.75, 13.11]
#
# # Create a blank image canvas
# width, height = 800, 600  # You can adjust the size
# image = Image.new("RGB", (width, height), (255, 255, 255))
# draw = ImageDraw.Draw(image)
#
# # Function to generate random shapes
# def random_shape(draw, color, area):
#     for _ in range(int(area * 1)):  # Increase number of shapes for finer control
#         x, y = random.randint(0, width), random.randint(0, height)
#         dx, dy = random.randint(10, 40), random.randint(10, 40)  # Smaller shapes
#         if random.choice([True, False]):
#             draw.ellipse((x - dx, y - dy, x + dx, y + dy), fill=color)
#         else:
#             draw.rectangle((x - dx, y - dy, x + dx, y + dy), fill=color)
#
# # Calculate total area and draw shapes for each color
# total_area = width * height / 1000  # Adjust for smaller shapes
# for color, percentage in zip(colors, percentages):
#     shape_area = total_area * (percentage / 100)
#     random_shape(draw, color, shape_area)
#
# # Save or display the image
# image.show()
# # image.save("multicam_camouflage.png")


import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('output30042024/1000_20240430_152344_image.png')

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Implement k-means clustering to segment the image
# We assume 5 clusters for the 5 main colors in the image
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)

# Map each pixel to the color of the centroid
segmented_image = centers[labels.flatten()]

# Reshape back to the original image dimensions
segmented_image = segmented_image.reshape(image.shape)
centers = [tuple(map(int, center)) for center in centers]

# Map each of the original colors to your new colors
mapped_colors = {
    centers[0]: (143, 171, 180),
    centers[1]: (31, 39, 43),
centers[2]: (96, 120, 127),
centers[3]: (206, 217, 220),
centers[4]: (58, 78, 82)
}

# Apply the mapping
for original_color, new_color in mapped_colors.items():
    mask = cv2.inRange(segmented_image, original_color, original_color)
    segmented_image[mask > 0] = new_color

# Convert back to uint8
segmented_image = np.uint8(segmented_image)

# Save or display the image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('new_multicam.jpg', segmented_image)
