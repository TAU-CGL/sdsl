import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def pgm_to_polygon(pgm_file, min_area=10):
    # Load the PGM file
    image = Image.open(pgm_file)
    image = np.array(image)
    image = image[:, :, 0]  # Extract single channel (assuming grayscale)

    # Flip y
    image = np.flipud(image)

    # Threshold the grid to binary (e.g., 0 for occupied, 1 for free space)
    # Assuming values > 200 are free space
    binary_grid = (image > 250).astype(np.uint8)

    # Find contours (boundaries) in the binary grid
    from skimage.measure import find_contours

    contours = find_contours(binary_grid, level=0.5)

    # Convert contours to polygons
    polygons = []
    for contour in contours:
        # Convert to (x, y) coordinates in image space
        coordinates = [(x, y) for y, x in contour]
        if len(coordinates) > 2:  # Valid polygon requires at least 3 points
            polygon = Polygon(coordinates)
            if polygon.area >= min_area:  # Filter small polygons by area
                simplified_polygon = polygon.simplify(10.0, preserve_topology=True)
                polygons.append(simplified_polygon)

    return polygons

# Example usage
if __name__ == "__main__":
    pgm_file = "map3.png"  # Replace with your PGM file
    polygons = pgm_to_polygon(pgm_file, min_area=100)  # Adjust min_area as needed

    # Display the polygon
    for poly in polygons:  # Use .geoms to iterate MultiPolygon
        x, y = poly.exterior.xy
        plt.plot(x, y)

    plt.gca().invert_yaxis()  # Invert Y-axis for image coordinate alignment
    plt.show()

    # Print polygon for verification
    poly_content = ""
    for poly in polygons:
        xs, ys = poly.exterior.xy
        for x, y in zip(xs[:-1], ys[:-1]):
            x = float(x) * 0.05 - 8.72
            y = float(y) * 0.05 - 115
            poly_content += f"{x} {y}\n"
        poly_content += "\n"
    with open("frog.poly", "w") as fp:
        fp.write(poly_content)
