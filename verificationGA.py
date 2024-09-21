# import random
# import numpy as np
# import cv2
# from deap import base, creator, tools, algorithms
#
# def extract_color_histogram(image_path, bins=(8, 8, 8)):
#     """Extract a color histogram from an image given its file path."""
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image at {image_path} could not be loaded. Check the file path and format.")
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist
#
# def calculate_fitness(features1, features2):
#     """Inverse of the chi-squared distance between histograms."""
#     distance = cv2.compareHist(features1, features2, cv2.HISTCMP_CHISQR)
#     return 1 / (1 + distance),
#
#
# import subprocess
#
#
# def convert_heic_to_png(heic_path, png_path):
#     # Form the command to convert the image
#     command = ['convert', heic_path, png_path]
#
#     # Execute the command
#     try:
#         subprocess.run(command, check=True)
#         print(f"Conversion successful: '{heic_path}' to '{png_path}'")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during conversion: {e}")
#
#
# # Example usage
# heic_path = 'output/IMG_2901.HEIC'  # Update this to your HEIC file's path
# png_path = 'naturephoto.png'  # Desired output path for the PNG file
# convert_heic_to_png(heic_path, png_path)
#
# # DEAP setup and definitions...
#
# # Load and extract features from the original pattern
# original_features = extract_color_histogram('output/1500_20240221_221456_image.png')
#
# # Assuming IMG_2901.HEIC has been converted to IMG_2901.jpg or any compatible format
# live_photo_path = 'naturephoto.png' # Update this path after conversion
# live_photo_features = extract_color_histogram(live_photo_path)
#
# def evalCamouflage(individual):
#     """Evaluate an individual by comparing live photo features to the original pattern."""
#     # This simulation just compares the live photo to the pattern directly for demonstration
#     # The individual could be used to alter parameters or methods of feature extraction
#     fitness = calculate_fitness(live_photo_features, original_features)
#     return fitness
#
# # DEAP toolbox setup including evaluation, mating, mutation, and selection strategies
# # Problem constants
# DIMENSIONS = 10  # Number of dimensions in the problem
# BOUND_LOW, BOUND_UP = -5.0, 5.0  # Bounds of the problem space
#
# # DEAP Framework setup
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
#
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
# toolbox.register("individual", tools.initIterate, creator.Individual,
#                  lambda: [toolbox.attr_float() for _ in range(DIMENSIONS)])
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# # Evaluation function
# def evaluate(individual):
#     """A simple evaluation function to demonstrate evolution."""
#     target = np.sin(1)  # Example dynamic component, could be iterated each generation
#     return sum((x - target) ** 2 for x in individual),
#
# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
#
# def main():
#     random.seed(64)
#     population = toolbox.population(n=300)
#
#     # Parameters
#     NGEN = 40
#     CXPB = 0.5
#     MUTPB = 0.3
#
#     # Statistics
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#
#     # Evolutionary algorithm
#     population, logbook = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats=stats, verbose=True)
#
#     return population, logbook
#
#
#
# if __name__ == "__main__":
#     main()


# import random
# from deap import base, creator, tools, algorithms
# import numpy as np
# import cv2
#
#
# # Placeholder function for image processing and similarity calculation
# def calculate_similarity(image1, image2):
#     # Here you would implement the actual similarity calculation
#     # For demonstration, let's return a random similarity
#     return random.random()
#
#
# # Define the individual (a set of transformations)
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
#
# # Initialize the toolbox
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, -1, 1)  # Example: scaling factors, rotation angles
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float,
#                  n=3)  # n is the number of transformations
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#
# # Define the evaluation function
# def evaluate(individual):
#     # Apply transformations based on the individual to the candidate image
#     # For demonstration, let's assume 'candidate_image' and 'reference_pattern' are defined
#     transformed_image = candidate_image  # Placeholder for the transformation logic
#     similarity = calculate_similarity(transformed_image, reference_pattern)
#     return similarity,
#
#
# # Register GA operations
# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# toolbox.register("select", tools.selTournament, tournsize=3)
#
#
# def main():
#     random.seed(64)
#     population = toolbox.population(n=50)
#
#     # GA parameters
#     NGEN = 40
#     CXPB = 0.5
#     MUTPB = 0.2
#
#     # Statistics
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#
#     # Run GA
#     population, log = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, stats, verbose=True)
#
#     return population, log
#
#
# if __name__ == "__main__":
#     # Placeholder image paths
#     candidate_image_path = "naturephoto.png"
#     reference_pattern_path = "output/1500_20240221_221456_image.png"
#
#     # Load images (you need to handle image loading and preprocessing)
#     candidate_image = cv2.imread(candidate_image_path)
#     reference_pattern = cv2.imread(reference_pattern_path)
#
#     main()

# import random
# import numpy as np
# import cv2
# from deap import base, creator, tools, algorithms
#
# def extract_edges(image1_path, image2_path):
#     """Extract edges from two images using the Canny edge detector, ensuring both images are compared at the same size."""
#     # Load images
#     image1 = image1_path
#     image2 = image2_path
#
#     # Check if images loaded successfully
#     if image1 is None or image2 is None:
#         raise ValueError("One of the images didn't load properly. Please check the paths.")
#
#     # Resize image2 to match image1's size
#     image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
#
#     # Convert images to grayscale
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
#     # Perform Canny edge detection
#     edges1 = cv2.Canny(gray1, 100, 200)
#     edges2 = cv2.Canny(gray2, 100, 200)
#
#     return edges1, edges2
#
# def calculate_similarity(edges1, edges2):
#     """Calculate the similarity between two sets of edges."""
#     # Compute the XOR to find differing pixels, then count the non-zero values (differences)
#     difference = cv2.bitwise_xor(edges1, edges2)
#     similarity = 1 - np.count_nonzero(difference) / difference.size
#     return similarity
#
# # GA setup
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
#
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, 0.9, 1.1)  # Scale factors around 1 (no change)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # Scale and threshold for edge detection
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# # Load the images
# image1 = cv2.imread('naturephoto.png')
# image2 = cv2.imread('output/1500_20240221_221456_image.png')
#
# edges1 = extract_edges(image1, image2)
#
# def evaluate(individual):
#     """Evaluate an individual's fitness by adjusting edge detection parameters and comparing."""
#     """Evaluate an individual's fitness by adjusting edge detection parameters and comparing."""
#     # Apply scaling (to simulate some basic image manipulation)
#     scaled_image2 = cv2.convertScaleAbs(image2, alpha=individual[0])
#
#     # Extract edges with resized dimensions
#     edges1, edges2 = extract_edges(image1, scaled_image2)
#
#     return calculate_similarity(edges1, edges2),
#
# # GA operators
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluate)
#
# def main():
#     random.seed(64)
#     pop = toolbox.population(n=100)
#     hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#
#     algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 40, stats=stats, halloffame=hof, verbose=True)
#
#     print("Best individual is:", hof[0], "with fitness:", hof[0].fitness.values[0])
#     return pop, stats, hof
#
# if __name__ == "__main__":
#     main()



import random
import numpy as np
import cv2
from deap import base, creator, tools, algorithms

def extract_edges(image_path, target_size=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image didn't load properly. Please check the path.")
    if target_size is not None:
        image = cv2.resize(image, target_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def calculate_diversity(edges1, edges2):
    difference = cv2.bitwise_xor(edges1, edges2)
    diversity_percentage = np.count_nonzero(difference) / difference.size * 100
    return diversity_percentage

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.0, 5.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

image1_path = 'lyagushka.jpg'
image2_path = 'output/1050_20240221_215425_image.png'
image1 = cv2.imread(image1_path)
target_size = (image1.shape[1], image1.shape[0])

edges1 = extract_edges(image1_path, target_size=target_size)
edges2 = extract_edges(image2_path, target_size=target_size)

def evaluate(individual):
    diversity = calculate_diversity(edges1, edges2)
    return diversity,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 30, stats=stats, halloffame=hof, verbose=True)
    best_individual = hof[0]
    best_diversity = calculate_diversity(edges1, edges2)
    print(f"Best individual is: {best_individual}, with diversity: {best_diversity}%")

if __name__ == "__main__":
    main()
