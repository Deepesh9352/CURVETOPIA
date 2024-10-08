import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Function to detect and process overlapping curves from CSV data
def detect_overlapping_curves_from_csv(path_XYs):
    output_image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White background

    for XYs in path_XYs:
        for XY in XYs:
            contour = XY.astype(np.int32).reshape((-1, 1, 2))
            regularized_contour, shape = classify_and_regularize(contour)
            
            if shape in ["Ellipse", "Circle"]:
                cv2.drawContours(output_image, [regularized_contour], -1, (0, 255, 0), 2)
            else:
                cv2.drawContours(output_image, [regularized_contour], -1, (0, 0, 0), 2)
    
    return output_image

# Main function to process the CSV file and generate the output image
def process_polylines_from_csv(csv_path, output_image_path='output_shape.png'):
    path_XYs = read_csv(csv_path)
    processed_image = detect_overlapping_curves_from_csv(path_XYs)
    
    # Save the processed image
    cv2.imwrite(output_image_path, processed_image)
    
    return processed_image

# Run the process and display the image
csv_input_path = r"C:\\Users\\DELL\\Downloads\\problems\\problems\\occlusion2.csv"
output_image_path = '/mnt/data/output_shape.png'
output_image = process_polylines_from_csv(csv_input_path, output_image_path)

# Convert BGR to RGB for displaying with Matplotlib
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(output_image_rgb)
plt.title('Processed Shape with Overlaps and Ellipses from CSV')
plt.axis('off')  # Hide axes
plt.show()



