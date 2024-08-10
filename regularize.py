import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        print("Loaded data:")
        print(np_path_XYs)
        
        path_XYs = []
        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []
            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY.astype(np.int32))
            path_XYs.append(XYs)
        
        return path_XYs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def detect_shapes_refined_from_csv(csv_path, output_path):
    path_XYs = read_csv(csv_path)
    if not path_XYs:
        return
    
    # Assuming a fixed size image (you may want to adjust this according to your needs)
    output = np.ones((500, 500, 3), dtype=np.uint8) * 255
    
    for path_contours in path_XYs:
        for contour in path_contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)

            # Filter out small contours
            if area < 100:
                continue

            # Draw the contours on the output image
            if len(approx) == 3:
                cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Triangle
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Square
                else:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rectangle
            elif len(approx) == 5:
                cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Pentagon
            elif len(approx) > 5:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                # Check if the contour is a circle by comparing area and perimeter
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.7 < circularity < 1.3:
                    cv2.circle(output, center, radius, (0, 0, 255), 2)  # Circle
                else:
                    cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # Other shapes

    # Save the output image
    cv2.imwrite(output_path, output)
    print(f"Output image saved at {output_path}")

# Define the CSV file path and output image path
csv_path = r"C:\\Users\\DELL\\Downloads\\problems\\problems\\frag2.csv"  # Replace with your CSV file path
refined_output_image_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\refined_output_image_from_csv.png"  # Replace with your output image path

# Detect shapes from the CSV data and save the output image
detect_shapes_refined_from_csv(csv_path, refined_output_image_path)

# Display the output image
output_image = Image.open(refined_output_image_path)
plt.imshow(output_image)
plt.axis('off')
plt.show()
