import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect edges with reduced noise
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise before edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Function to classify and regularize shapes
def classify_and_regularize(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    shape = "Unknown"
    regularized_contour = approx

    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        regularized_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    elif len(approx) > 10:
        area = cv2.contourArea(contour)
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        circularity = area / (np.pi * (MA / 2) * (ma / 2))
        if circularity > 0.85:
            if 0.9 <= MA / ma <= 1.1:
                shape = "Circle"
                regularized_contour = cv2.ellipse2Poly((int(x), int(y)), (int(MA / 2), int(MA / 2)), 0, 0, 360, 1)
            else:
                shape = "Ellipse"
                regularized_contour = cv2.ellipse2Poly((int(x), int(y)), (int(MA / 2), int(ma / 2)), int(angle), 0, 360, 1)
        else:
            shape = "Complex"
            regularized_contour = approx
    else:
        shape = "Polygon"
    
    return regularized_contour, shape

# Function to complete shapes using convex hull
def complete_shape(contour):
    return cv2.convexHull(contour)

# Function to read CSV file and convert to polylines
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Main function to process the polylines from CSV file
def process_polylines(csv_path, output_image_path='output_shape.png'):
    path_XYs = read_csv(csv_path)
    
    # Create a white background (assuming a fixed size, adjust as needed)
    output_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    for XYs in path_XYs:
        for XY in XYs:
            # Convert polyline to a contour format
            contour = XY.astype(np.int32).reshape((-1, 1, 2))
            regularized_contour, _ = classify_and_regularize(contour)
            completed_contour = complete_shape(regularized_contour)
            cv2.drawContours(output_image, [completed_contour], -1, (0, 0, 0), 2)
    
    # Save the processed image
    cv2.imwrite(output_image_path, output_image)
    
    return output_image

# Run the process and display the image
output_image = process_polylines(r"C:\\Users\\DELL\\Downloads\\problems\\problems\\frag0.csv")

# Convert BGR to RGB for displaying with Matplotlib
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(output_image_rgb)
plt.title('Processed Shape')
plt.axis('off')  # Hide axes
plt.show()
