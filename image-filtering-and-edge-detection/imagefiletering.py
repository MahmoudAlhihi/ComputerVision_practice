import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read the image 
img = cv2.imread('cityscape.jpg', cv2.IMREAD_COLOR)
# Convert the image from BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter (it reduces noise and smooths the image)
# grayscale input image, 5x5 kernel size and 0 standard deviation
gaussian = cv2.GaussianBlur(gray,(5,5) , 0) 
gaussian2 = cv2.GaussianBlur(gray,(9,9) , 0) # stronger blur, more smoothing but slower
gaussian3 = cv2.GaussianBlur(gray,(3,3) , 0) # weaker blur, preserves more details

# Apply Sobel Operator
# Sobel X detects vertical edges and Sobel Y detects horizontal edges

sobelx = cv2.Sobel(gaussian, cv2.CV_64F, 1 , 0 ,ksize=5)
sobely = cv2.Sobel(gaussian, cv2.CV_64F, 0 , 1 ,ksize=5)

sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = cv2.convertScaleAbs(sobel_combined)


# Apply Canny Edge Detection
blur = cv2.GaussianBlur(gray, (5,5), 1.4) # Remove small pixel flucations that would become false edges
edges = cv2.Canny(blur,100, 200 ) #Lower and upper threshold

#Visualizing
plt.figure(figsize=(15,6))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(gaussian, cmap ='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(sobel_combined, cmap ='gray')
plt.title('Sobel Edges')
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(edges, cmap ='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.tight_layout()
plt.show()