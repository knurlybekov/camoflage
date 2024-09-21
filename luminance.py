from PIL import Image
import numpy as np


def segment_and_save_image(original_image_path, num_segments=9):
    image = Image.open(original_image_path)
    width, height = image.size
    segment_width = width // 3
    segment_height = height // 3
    segments = []

    for i in range(3):
        for j in range(3):
            box = (j * segment_width, i * segment_height, (j + 1) * segment_width, (i + 1) * segment_height)
            segment = image.crop(box)
            segment_path = f'region_{i * 3 + j}.jpg'
            segment.save(segment_path)
            segments.append(segment_path)

    return segments


def equalize_histogram(image):
    np_image = np.array(image)
    hist, bins = np.histogram(np_image.flatten(), 256, density=True)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    image_equalized = np.interp(np_image.flatten(), bins[:-1], cdf)
    image_equalized = image_equalized.reshape(np_image.shape).astype(np.uint8)
    return Image.fromarray(image_equalized)


def process_images(image_paths):
    images = [Image.open(path) for path in image_paths]
    grayscale_images = [img.convert('L') for img in images]
    luminance_arrays = [np.array(img) for img in grayscale_images]
    average_luminance_array = np.mean(luminance_arrays, axis=0)
    average_luminance_image = Image.fromarray(average_luminance_array.astype(np.uint8))
    flattened_luminance_image = equalize_histogram(average_luminance_image)
    return flattened_luminance_image


# Segment and save parts of the original image
original_image_path = 'kamloops.jpg'  # Specify your original image path here
segment_paths = segment_and_save_image(original_image_path)

# Process the saved segments
flattened_image = process_images(segment_paths)

# Save the final image
flattened_image.save('flattened_luminance_image.jpg')
flattened_image.show()
