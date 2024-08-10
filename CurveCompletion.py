import csv
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

# Function to read CSV and parse polylines
def read_csv(filepath):
    polylines = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            polyline = [(float(x), float(y)) for x, y in zip(row[::2], row[1::2])]
            polylines.append(polyline)
    return polylines

# Function to write polylines to CSV
def write_csv(polylines, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for polyline in polylines:
            row = [coord for point in polyline for coord in point]
            writer.writerow(row)

# Helper function to check if a polyline is a straight line
def is_straight_line(polyline):
    if len(polyline) < 3:
        return True
    direction = np.diff(polyline, axis=0)
    direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
    return np.allclose(direction, direction[0])

# Helper function to fit an ellipse to a set of points
def fit_ellipse(polyline):
    x = np.array([p[0] for p in polyline])
    y = np.array([p[1] for p in polyline])
    x_m = np.mean(x)
    y_m = np.mean(y)
    x -= x_m
    y -= y_m
    U, S, V = np.linalg.svd(np.stack([x, y]))
    radii = np.sqrt(2) * np.std(U @ np.diag(S))
    theta = np.arctan2(V[0, 1], V[0, 0])
    return (x_m, y_m), radii, theta

# Helper function to fit a circle to a set of points
def fit_circle(polyline):
    def calc_R(xc, yc):
        return np.sqrt((polyline[:, 0] - xc) ** 2 + (polyline[:, 1] - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    polyline = np.array(polyline)
    x_m, y_m = np.mean(polyline, axis=0)
    center_estimate = x_m, y_m
    center, _ = minimize(f_2, center_estimate).x, minimize(f_2, center_estimate).fun
    Ri = calc_R(*center)
    return center, Ri.mean()

# Function to classify shapes
def classify_shapes(polylines):
    classified_shapes = []
    for polyline in polylines:
        if is_straight_line(polyline):
            classified_shapes.append(('line', polyline))
        elif len(polyline) > 3:
            center, radius = fit_circle(polyline)
            if np.std([np.linalg.norm(np.array(p) - center) for p in polyline]) < 0.1:
                classified_shapes.append(('circle', polyline, center, radius))
            else:
                center, radii, theta = fit_ellipse(polyline)
                if radii[0] / radii[1] > 1.1:
                    classified_shapes.append(('ellipse', polyline, center, radii, theta))
                else:
                    classified_shapes.append(('polygon', polyline))
        else:
            classified_shapes.append(('polygon', polyline))
    return classified_shapes

# Function to detect incomplete shapes
def detect_incomplete_shapes(shapes):
    incomplete_shapes = []
    for shape in shapes:
        if shape[0] == 'line':
            incomplete_shapes.append(shape)
        elif shape[0] == 'circle':
            if len(shape[1]) < 20:
                incomplete_shapes.append(shape)
        elif shape[0] == 'ellipse':
            if len(shape[1]) < 20:
                incomplete_shapes.append(shape)
        elif shape[0] == 'polygon':
            if ConvexHull(shape[1]).volume < 0.5:
                incomplete_shapes.append(shape)
    return incomplete_shapes

# Function to complete shapes
def complete_shapes(incomplete_shapes):
    completed_shapes = []
    for shape in incomplete_shapes:
        if shape[0] == 'line':
            completed_shapes.append(shape)
        elif shape[0] == 'circle':
            center, radius = shape[2], shape[3]
            num_missing_points = 100 - len(shape[1])
            angles = np.linspace(0, 2 * np.pi, num_missing_points)
            missing_points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
            completed_shapes.append(('circle', shape[1] + missing_points, center, radius))
        elif shape[0] == 'ellipse':
            center, radii, theta = shape[2], shape[3], shape[4]
            num_missing_points = 100 - len(shape[1])
            angles = np.linspace(0, 2 * np.pi, num_missing_points)
            missing_points = [(center[0] + radii[0] * np.cos(a) * np.cos(theta) - radii[1] * np.sin(a) * np.sin(theta),
                               center[1] + radii[0] * np.cos(a) * np.sin(theta) + radii[1] * np.sin(a) * np.cos(theta)) for a in angles]
            completed_shapes.append(('ellipse', shape[1] + missing_points, center, radii, theta))
        elif shape[0] == 'polygon':
            completed_shapes.append(shape)
    return completed_shapes

# Function to resolve overlaps
def resolve_overlaps(shapes):
    resolved_shapes = []
    for shape in shapes:
        resolved_shapes.append(shape)
    return resolved_shapes

# Function to reconstruct polylines
def reconstruct_polylines(resolved_shapes):
    polylines = []
    for shape in resolved_shapes:
        polylines.append(shape[1])
    return polylines

# Function to detect if a polygon is a rectangle
def is_rectangle(polyline):
    if len(polyline) != 4:
        return False
    poly = Polygon(polyline)
    return poly.is_valid and poly.is_convex

# Function to detect if a polygon is a star
def is_star(polyline):
    if len(polyline) < 6:
        return False
    poly = Polygon(polyline)
    return poly.is_valid and not poly.is_convex

# Function to classify rectangles, stars, and polygons
def classify_rectangles_stars(polylines):
    classified_shapes = []
    for polyline in polylines:
        if len(polyline) < 4:
            continue  # Skip polylines that are too short
        if is_rectangle(polyline):
            classified_shapes.append(('rectangle', polyline))
        elif is_star(polyline):
            classified_shapes.append(('star', polyline))
        else:
            classified_shapes.append(('polygon', polyline))
    return classified_shapes

# Main function to complete polylines
def complete_polylines(input_csv, output_csv):
    # Step 1: Input Parsing
    polylines = read_csv(input_csv)
    
    # Step 2: Shape Detection and Classification
    classified_shapes = classify_shapes(polylines)
    classified_shapes += classify_rectangles_stars(polylines)
    
    # Step 3: Incomplete Shape Detection
    incomplete_shapes = detect_incomplete_shapes(classified_shapes)
    
    # Step 4: Shape Completion
    completed_shapes = complete_shapes(incomplete_shapes)
    
    # Step 5: Overlap Resolution
    resolved_shapes = resolve_overlaps(completed_shapes)
    
    # Step 6: Final Polyline Reconstruction
    final_polylines = reconstruct_polylines(resolved_shapes)
    
    # Step 7: Output Generation
    write_csv(final_polylines, output_csv)

# Example usage
input_csv = r"C:\\Users\\DELL\\Downloads\\problems\\problems\\occlusion1.csv"
output_csv = r"C:\\Users\\DELL\\Downloads\\problems\\problems\\output_polylines.csv"
complete_polylines(input_csv, output_csv)
