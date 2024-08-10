import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def detect_edges(binary_image):
    # Detect edges using Canny edge detector
    edges = cv2.Canny(binary_image, 50, 150)
    return edges

def find_contours(edges):
    # Find contours in the binary image and retrieve hierarchy to handle nested shapes
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def approximate_contours(contours, epsilon_factor=0.005):
    approx_contours = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)
    return approx_contours

def fit_bezier_curves(approx_contours):
    # Convert contours to cubic Bézier curves (using placeholder functions)
    bezier_curves = []
    for contour in approx_contours:
        # Fit cubic Bézier curves to each contour (using a placeholder function)
        bezier_curves.append(contour_to_bezier(contour))
    return bezier_curves

def contour_to_bezier(contour):
    # Placeholder function to convert contour to Bézier curves
    # You can replace this with an actual implementation
    return contour

def save_as_svg(bezier_curves, output_path, image_shape):
    dwg = svgwrite.Drawing(output_path, profile='tiny', size=(image_shape[1], image_shape[0]))
    for curve in bezier_curves:
        # Convert each Bézier curve to an SVG path (using placeholder functions)
        path_data = bezier_to_svg_path(curve)
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=2))
    dwg.save()

def bezier_to_svg_path(bezier_curve):
    # Placeholder function to convert Bézier curve to SVG path data
    # You can replace this with an actual implementation
    return 'M 0 0 L 100 100'

def display_results(approx_contours, image_shape, hierarchy):
    # Create a white canvas
    approx_image = np.ones(image_shape, dtype=np.uint8) * 255
    
    for i, contour in enumerate(approx_contours):
        # Draw the contours based on hierarchy information to manage nested shapes
        color = (0, 0, 0) if hierarchy[0][i][3] == -1 else (150, 150, 150)
        cv2.drawContours(approx_image, [contour], -1, (0, 0, 0), 2)
    
    plt.figure(figsize=(8, 8))
    plt.title("Approximate Contours")
    plt.imshow(approx_image, cmap='gray')
    plt.axis('off')
    plt.show()

def main(image_path, output_path):
    image = load_image(image_path)
    binary_image = preprocess_image(image)
    edges = detect_edges(binary_image)
    contours, hierarchy = find_contours(edges)
    approx_contours = approximate_contours(contours)
    bezier_curves = fit_bezier_curves(approx_contours)
    save_as_svg(bezier_curves, output_path, image.shape)
    display_results(approx_contours, image.shape, hierarchy)

if __name__ == "__main__":
    input_image_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\frag0.png"
    output_svg_path = "output.svg"
    main(input_image_path, output_svg_path)
