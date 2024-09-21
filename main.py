import numpy as np
import cv2
import sys

def std_calc(img, w):                             #function to calculate std values for each pixel location
    stdMatrix = np.zeros(img.shape,np.uint8)
    ny = len(img)
    nx = len(img[0])


    for i in range(w,nx-w):
        for j in range(w,ny-w):
            sampleframe = img[j-w:j+w, i-w:i+w]
            std = np.std(sampleframe)
            stdMatrix[j][i] = int(std)
    return stdMatrix

if len(sys.argv) < 5:
    print("Error: Insufficient arguments, program takes four additional arguments : \n 1. input image name\n 2. K\n 3. W : dimension of filter for std dev\n 4. option for feature selection(X,Y,std/texture) : \n|X  |Y  |V  |\n|0/1|0/1|0/1|\n Eg : python kmeans_rgb_var.py test2.jpeg 5 3 101")
    sys.exit()
else:
    file = sys.argv[1]
    K = int(sys.argv[2])
    if K < 3:
        print("Error: K has to be greater than 2")
        sys.exit()
    w = int(sys.argv[3])  #dimension of filter for std dev
    if w < 1:
        print("Error: w has to be greater than 0")
        sys.exit()
    option = sys.argv[4]
    if(len(option) != 3):
        print("Error:Std option has to be of length 3")
        sys.exit()


img = cv2.imread('pine-timber-guide-0.jpg') #read image
Z = np.float32(img)
flag = 0
stdMatrix = std_calc(Z,w)   #calculate std dev values
flat_std = np.zeros((img.shape[0]*img.shape[1],1))
flat_x = np.zeros((img.shape[0]*img.shape[1],1))
flat_y = np.zeros((img.shape[0]*img.shape[1],1))

for i in range(stdMatrix.shape[0]):                   #flatten the std dev matrix. It has the same shape as image. but std dev values for r,g,b are same
    for j in range(stdMatrix.shape[1]):
        flat_std[flag] = [stdMatrix[i][j][0]]
        flag += 1


flag=0
for i in range(stdMatrix.shape[0]):                   #flatten the x-values matrix. It has the same shape as image. but std dev values for r,g,b are same
    for j in range(stdMatrix.shape[1]):
        flat_x[flag] = [i]
        flag += 1


flag = 0
for i in range(stdMatrix.shape[0]):                   #flatten the y-values matrix. It has the same shape as image. but std dev values for r,g,b are same
    for j in range(stdMatrix.shape[1]):
        flat_y[flag] = [j]
        flag += 1




'''  shows that all three values of std dev matrix corresponding to a pixel location are the same
print(stdMatrix.shape)
print(stdMatrix[100][32].shape,stdMatrix[100][32])
print("Z : 1 : ",Z.shape)
for i in range(stdMatrix.shape[0]):
    for j in range(stdMatrix.shape[1]):
        if stdMatrix[i][j][0] != stdMatrix[i][j][1] != stdMatrix[i][j][2]:
            print(stdMatrix[i][j],i,j)
            flag = 1

if flag == 1:
    print("instance")
'''




Z2 = Z.reshape((-1,3))  #flattens the rgb features dim : (h*w,3)


# concat x/y/standard deviaiton features as requested by user to the rgb feature array
if(option[0] == '1'):
    Z2 = np.concatenate((Z2,np.float32(flat_x)),axis = 1)
if(option[1] == '1'):
    Z2 = np.concatenate((Z2,np.float32(flat_y)),axis = 1)
if(option[2] == '1'):
    Z2 = np.concatenate((Z2,np.float32(flat_std)),axis = 1)




# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print("Calculting Centers ......\n")
ret,label,center=cv2.kmeans(Z2,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print("Center matrix : \n",center)
# Now convert back into uint8, and make original image
center = np.uint8(np.hsplit(center,np.array([3,4]))[0])   #remove x/y/std values from data

#print("Centers\n : ",center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))


cv2.imwrite(sys.argv[0]+"out(rgb)_"+str(K)+"_"+file, res2)



##################################################################################################
########################################################################################################
################################################################################################
# Comparing with Lab color space


Z1 = Z/255.0

Z2 = cv2.cvtColor(Z1,44)
Z3 = Z2.reshape((-1,3))
#print(Z3.shape)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print("Calculting Centers for Lab space ......\n")
ret,label,center=cv2.kmeans(Z3,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)  #distance function for CIE Lab : E76 similar to euclidean distance


print("Centers in Lab space : \n",center)
center2 = (cv2.cvtColor(center.reshape((K,1,3)),56))*255.0  #convert center coordinates to rgb
print("Centers in rgb space based on lab k-means : \n",center2)
# Now convert back into uint8, and make original image
center3 = np.uint8(center2)
res = center3[label.flatten()]

#print(res.shape)
res2 = res.reshape((img.shape))
#print(res2.shape)


cv2.imwrite(sys.argv[0]+"out(lab)_"+str(K)+"_"+file, res2)

# # Read batman image and print dimensions
# batman_image = img.imread('batman.png')
#
# # Store RGB values of all pixels in lists r, g and b
# r = []
# g = []
# b = []
# for row in batman_image:
#     for temp_r, temp_g, temp_b, temp in row:
#         r.append(temp_r)
#         g.append(temp_g)
#         b.append(temp_b)
#
# # only printing the size of these lists
# # as the content is too big
# print(len(r))
# print(len(g))
# print(len(b))
#
# # Saving as DataFrame
# batman_df = pd.DataFrame({'red': r,
#                           'green': g,
#                           'blue': b})
#
# # Scaling the values
# batman_df['scaled_color_red'] = whiten(batman_df['red'])
# batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
# batman_df['scaled_color_green'] = whiten(batman_df['green'])


# image = cv2.imread('pine-timber-guide-0.jpg')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Example: Canny edge detection
# edges = cv2.Canny(gray_image, 50, 150)
# # Example: Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Iterate through contours and do further analysis
# for contour in contours:
#     # Your analysis code here
#
#     cv2.imshow('Original Image', image)
#
#     # Display the processed image
#     cv2.imshow('Processed Image', edges)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# # Convert the image to the HSV color space
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # Define the lower and upper bounds for the green and brown colors typically found in woods
# lower_green = np.array([40, 40, 40])
# upper_green = np.array([80, 255, 255])
#
# lower_brown = np.array([10, 50, 50])
# upper_brown = np.array([20, 255, 255])
#
# # Create masks for green and brown colors
# mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
# mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
#
# # Combine the masks to get the final camouflage mask
# camouflage_mask = cv2.bitwise_or(mask_green, mask_brown)
#
# # Apply the mask to the original image
# result = cv2.bitwise_and(image, image, mask=camouflage_mask)
#
# # Display the original image and the result
# cv2.imshow('Original Image', image)
# cv2.imshow('Camouflage Pattern', result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def get_pattern(image):
    # Load an image
    img = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Preprocess the image
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the template (target shape)
    template = cv2.imread('template.jpg', 0)
    if template is None:
        print("Error loading template image")
        return

    # Similar preprocessing for the template
    _, thresh_template = cv2.threshold(template, 240, 255, cv2.THRESH_BINARY)
    contours_template, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_template:
        target_contour = contours_template[0]
    else:
        print("No contours found in template image.")
        return

    # Iterate over contours found in the nature photo
    for contour in contours:
        # Compare shapes
        similarity = cv2.matchShapes(target_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)

        # If the shapes are similar (similarity value is low), draw it on the original image
        if similarity < 0.1:  # threshold value is adjustable
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

    # Display the result
    cv2.imshow('Matched Shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Example: Canny edge detection
    # edges = cv2.Canny(gray_image, 50, 150)
    # # Example: Find contours
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Iterate through contours and do further analysis
    # for contour in contours:
    #     # Your analysis code here
    #
    #     cv2.imshow('Original Image', image)
    #
    #     # Display the processed image
    #     cv2.imshow('Processed Image', edges)
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # # Convert the image to the HSV color space
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # # Define the lower and upper bounds for the green and brown colors typically found in woods
    # lower_green = np.array([40, 40, 40])
    # upper_green = np.array([80, 255, 255])
    #
    # lower_brown = np.array([10, 50, 50])
    # upper_brown = np.array([20, 255, 255])
    #
    # # Create masks for green and brown colors
    # mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    # mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    #
    # # Combine the masks to get the final camouflage mask
    # camouflage_mask = cv2.bitwise_or(mask_green, mask_brown)
    #
    # # Apply the mask to the original image
    # result = cv2.bitwise_and(image, image, mask=camouflage_mask)
    #
    # # Display the original image and the result
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Camouflage Pattern', result)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def get_colors(image):
    # image = cv2.imread(image)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image to a 1D array
    pixels = image_rgb.reshape((-1, 3))

    # Calculate the unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Calculate the total number of pixels
    total_pixels = np.sum(counts)

    # Calculate the percentage of each color
    percentages = (counts / total_pixels) * 100

    # Combine colors, counts, and percentages into a list of tuples
    color_info = list(zip(unique_colors, counts, percentages))
    for color, count, percentage in color_info:
        print(f"Color: {color}, Count: {count}, Percentage: {percentage:.10f}%")


# get_colors(image)

def simplify_image(image_path, k=8):
    # Load the image
    image = image_path

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert pixel values to float
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)

    # Display the original and segmented images
    cv2.imshow('Original Image', image)
    cv2.imshow(f'Simplified Image (k={k})', segmented_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return segmented_image

# # Example usage
# image_path = image
# # # simplify_image(image_path, k=8)
# get_colors(simplify_image(image_path, k=8))


# get_pattern(image)

# Load the nature image
# nature_image = cv2.imread('pine-timber-guide-0.jpg')
# nature_image = cv2.resize(nature_image, (1500, 951))  # Resize to match multicam pattern size
#
# # Load the multicam pattern
# multicam_pattern = cv2.imread('multicam.jpg')
# multicam_pattern = cv2.resize(multicam_pattern, (1500, 951))
#
# # Blend the images
# alpha = 0.5  # blending ratio
# combined_pattern = cv2.addWeighted(nature_image, alpha, multicam_pattern, 1 - alpha, 0)
#
# # Save or display the result
# cv2.imshow('Combined Pattern', combined_pattern)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

