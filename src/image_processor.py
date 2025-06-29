import cv2
import numpy as np
import json
from sklearn.cluster import DBSCAN


class ImageProcessor:
    """
    A class to process an image, extract its features, and export the results.
    """

    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with the path to the image.

        Args:
            image_path (str): Path to the input image.
        """
        self.image_path = image_path
        self.image = None  # Original image
        self.gray_image = None  # Grayscale image
        self.masked_image = None  # Preprocessed grayscale image with white regions excluded
        self.data = {}

    def load_image(self):
        """
        Load the image in different formats: original, grayscale.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def preprocess_image(self):
        """
        Preprocess the image to mask and exclude white regions.
        Updates:
            self.masked_image: Modified grayscale image with white regions masked out.
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define a white mask range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Invert the mask to focus on non-white areas
        non_white_mask = cv2.bitwise_not(white_mask)

        # Apply the mask to the grayscale image
        self.masked_image = cv2.bitwise_and(
            self.gray_image, self.gray_image, mask=non_white_mask)

    def draw_contours(self, output_path="contours_output.jpg", blur_ksize=5, canny_threshold1=35, canny_threshold2=90, use_adaptive_threshold=False):
        """
        Draw contours with both green and white versions.

        Returns:
            list: Detected contours.
        """
        if self.masked_image is None:
            raise ValueError("Masked image not available. Preprocess the image first.")

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(self.masked_image, (blur_ksize, blur_ksize), 0)

        # Choose thresholding method
        if use_adaptive_threshold:
            edges = cv2.adaptiveThreshold(
                blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # --- GREEN version ---
        image_green = self.image.copy()
        cv2.drawContours(image_green, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(output_path, image_green)
        print(f"Contours (green) saved to {output_path}")

        # --- WHITE version ---
        image_white = self.image.copy()
        cv2.drawContours(image_white, contours, -1, (255, 255, 255), 1)
        white_output_path = output_path.replace("_contours_", "_contoursWHITE_")
        cv2.imwrite(white_output_path, image_white)
        print(f"Contours (white) saved to {white_output_path}")

        return contours

    def draw_contours_on_black(self, output_path="contours_on_black.png", blur_ksize=5, canny_threshold1=35, canny_threshold2=90, use_adaptive_threshold=False):
        """
        Draw contours on a black background, both in green and white.

        Args:
            output_path (str): Path to save the image with contours (green).
            blur_ksize (int): Kernel size for Gaussian blur.
            canny_threshold1 (int): Lower Canny threshold.
            canny_threshold2 (int): Upper Canny threshold.
            use_adaptive_threshold (bool): Whether to use adaptive thresholding instead of Canny.
        """
        if self.masked_image is None:
            raise ValueError("Masked image not available. Preprocess the image first.")

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(self.masked_image, (blur_ksize, blur_ksize), 0)

        # Detect edges
        if use_adaptive_threshold:
            edges = cv2.adaptiveThreshold(
                blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # --- GREEN version on black background ---
        background_green = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
        cv2.drawContours(background_green, contours, -1, (0, 255, 0, 255), 1)
        cv2.imwrite(output_path, background_green)
        print(f"Contours (green) on black saved to {output_path}")

        # --- WHITE version on black background ---
        background_white = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
        cv2.drawContours(background_white, contours, -1, (255, 255, 255, 255), 1)
        white_output_path = output_path.replace("_contoursBLACK_", "_contoursWHITEBLACK_")
        cv2.imwrite(white_output_path, background_white)
        print(f"Contours (white) on black saved to {white_output_path}")


    def detect_clusters(self, blur_ksize=5, canny_threshold1=35, canny_threshold2=90, use_adaptive_threshold=False):
        """
        Detect clusters based on contours and map X positions to a 1-minute timeline.
        Adds detailed cluster data to self.data, including average luminance per cluster.
        """
        # Get contours from draw_contours
        contours = self.draw_contours(
            blur_ksize=blur_ksize,
            canny_threshold1=canny_threshold1,
            canny_threshold2=canny_threshold2,
            use_adaptive_threshold=use_adaptive_threshold
        )

        # Calculate total size of contours
        total_contour_size = sum(cv2.contourArea(contour)
                                 for contour in contours)
        self.data["total_contour_size"] = total_contour_size
        print(f"Total contour size: {total_contour_size}")

        # Extract coordinates of the contours
        points = []
        for contour in contours:
            for point in contour:
                points.append(point[0])

        points = np.array(points)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=50, min_samples=150).fit(points)
        labels = clustering.labels_

        clusters = []
        height, width = self.image.shape[:2]

        # Convert the grayscale image to a NumPy array for faster indexing
        grayscale_array = np.asarray(self.gray_image)

        # Process each cluster
        for label in set(labels):
            if label == -1:
                continue  # Skip noise points

            # Points in the current cluster
            cluster_points = points[labels == label]
            avg_x = np.mean(cluster_points[:, 0])
            avg_y = np.mean(cluster_points[:, 1])

            # Map X position to time on a 1-minute track
            normalized_x = avg_x / width  # Normalize X to [0, 1]
            time_in_seconds = normalized_x * 60  # Map to 0-60 seconds

            # Calculate size and weight
            size = len(cluster_points)  # Number of points in the cluster
            hull = cv2.convexHull(cluster_points)
            # Approximate surface area of the cluster
            weight = cv2.contourArea(hull)

            # Calculate average luminance of the cluster
            luminance_values = []
            for point in cluster_points:
                x, y = int(point[0]), int(point[1])  # Ensure integer indices
                luminance_values.append(grayscale_array[y, x])

            avg_luminance = np.mean(
                luminance_values) if luminance_values else 0

            avg_color = [0, 0, 0, 0]
            if len(cluster_points) > 0:
                color_values = [
                    # Get BGR color at this point
                    self.image[int(point[1]), int(point[0])]
                    for point in cluster_points
                    if 0 <= int(point[1]) < height and 0 <= int(point[0]) < width
                ]
                avg_color = np.mean(
                    color_values, axis=0) if color_values else avg_color
                # Ensure alpha = 255 (fully opaque)
                avg_color = np.append(avg_color, 255)

            # Add cluster information
            clusters.append({
                "x": float(avg_x),
                "y": float(avg_y),
                "time": float(time_in_seconds),
                "size": size,
                "weight": weight,
                "avg_luminance": float(avg_luminance),
                "avg_color": avg_color.astype(int).tolist()

            })

        self.data["clusters"] = clusters

    def compute_attributes(self):
        """
        Compute attributes of the image such as size, average luminance, and contour density.
        Returns:
            dict: A dictionary containing image attributes and contour-based nodes.
        """
        # Load and preprocess the image
        self.load_image()
        self.preprocess_image()

        # Dimensions of the image
        height, width, channels = self.image.shape
        self.data["dimensions"] = {
            "height": height,
            "width": width,
            "channels": channels
        }

        # Compute luminance using the standard formula
        luminance_image = (0.2126 * self.image[:, :, 2] +
                           0.7152 * self.image[:, :, 1] +
                           0.0722 * self.image[:, :, 0])
        avg_luminance = np.mean(luminance_image)
        self.data["avg_luminance"] = avg_luminance

        return self.data

    def export_json(self, output_path):
        """
        Export the computed data to a JSON file.

        Args:
            output_path (str): Path to save the JSON file.
        """
        def convert_to_serializable(obj):
            """
            Recursively convert numpy types to native Python types.
            """
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj  # Return as-is if it's already a serializable type

        # Convert the data
        serializable_data = convert_to_serializable(self.data)

        # Write to JSON
        with open(output_path, "w") as json_file:
            json.dump(serializable_data, json_file, indent=4)

# Usage example:
# processor = ImageProcessor("path_to_image.jpg")
# processor.compute_attributes()
# processor.detect_clusters()
# processor.export_json("output.json")
