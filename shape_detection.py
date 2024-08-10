import numpy as np
import cv2
import os

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

def classify_shape(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 2:
        return "Straight line"
    elif num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Rounded rectangle"
        else:
            return "Rectangle"
    elif num_vertices > 6:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.85:
            return "Circle"
        elif 0.5 < circularity < 0.85:
            return "Ellipse"
        else:
            return "Star shape"
    elif 5 <= num_vertices <= 6:
        return "Regular Polygon"
    
    return "Unknown Shape"

def detect_shapes_from_csv(csv_path):
    path_XYs = read_csv(csv_path)
    if not path_XYs:
        return
    
    for path_contours in path_XYs:
        for contour in path_contours:
            shape = classify_shape(contour)
            print(f"Detected shape: {shape}")

def main():
    csv_path = r"C:\\Users\\DELL\\Downloads\\problems\\problems\\frag0.csv"
    
    if os.path.exists(csv_path):
        print(f"Processing CSV file at {csv_path}.")
        detect_shapes_from_csv(csv_path)
    else:
        print(f"The CSV file does not exist at {csv_path}.")

if __name__ == "__main__":  # Corrected this line
    main()

