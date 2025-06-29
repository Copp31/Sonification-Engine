import networkx as nx
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class GraphBuilder:
    """
    A class to build, manipulate, and visualize a graph from image data.
    """

    def __init__(self):
        """
        Initialize the GraphBuilder with an empty graph.
        """
        self.graph = nx.Graph()

    def add_nodes(self, clusters, dimensions):
        """
        Add nodes to the graph based on detected clusters.

        Args:
            clusters (list): List of clusters, where each cluster contains attributes like x, y, size, and avg_luminance.
            dimensions (dict): Dictionary containing image dimensions (width, height).
        """
        # Add global dimensions to the graph as attributes
        self.graph.graph['width'] = dimensions['width']
        self.graph.graph['height'] = dimensions['height']

        # Add individual nodes
        for idx, cluster in enumerate(clusters):
            # Extract position, size, and avg_luminance from the cluster
            position = (cluster['x'], cluster['y'])  # (x, y)
            size = cluster.get('size', 1)  # Default size is 1
            avg_luminance = cluster.get(
                'avg_luminance', 0)  # Default luminance is 0

            # Add node to the graph
            self.graph.add_node(
                idx,  # Use the index as the node ID
                position=position,
                size=size,
                avg_luminance=avg_luminance
            )

    def add_edges(self, max_distance=1000):
        """
        Add edges between nodes dynamically based on proximity.

        Args:
            max_distance (float): Maximum distance between nodes to create an edge.
        """
        nodes = list(self.graph.nodes)

        # Iterate over all pairs of nodes
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                pos_a = self.graph.nodes[node_a]['position']
                pos_b = self.graph.nodes[node_b]['position']

                # Calculate Euclidean distance between nodes
                distance = np.linalg.norm(np.array(pos_a) - np.array(pos_b))

                # Add edge if distance is below the threshold
                if distance <= max_distance:
                    self.graph.add_edge(
                        node_a,
                        node_b,
                        weight=distance,  # Use distance as the weight
                        distance=distance  # Explicit distance attribute
                    )

    def compute_weights(self):
        """
        Compute and update weights for all edges based on node attributes.

        This function calculates edge weights based on distances between node positions.
        """
        for edge in self.graph.edges:
            node_a, node_b = edge
            pos_a = self.graph.nodes[node_a]['position']
            pos_b = self.graph.nodes[node_b]['position']

            # Euclidean distance as the weight
            distance = ((pos_a[0] - pos_b[0])**2 +
                        (pos_a[1] - pos_b[1])**2)**0.5
            self.graph.edges[node_a, node_b]['weight'] = distance

    def export_graph(self, output_path):
        """
        Export the graph as a JSON file.

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

        # Convert the graph data to a node-link format
        graph_data = nx.node_link_data(self.graph, edges="edges")

        # Convert graph data to a serializable format
        serializable_data = convert_to_serializable(graph_data)

        # Write to JSON file
        try:
            with open(output_path, "w") as json_file:
                json.dump(serializable_data, json_file, indent=4)
        except Exception as e:
            print(f"Error exporting graph to JSON: {e}")

    def visualize_graph(self, output_path="graph_visualization.png", image_dimensions=None):
        """
        Visualize the graph on a white background with black edges and labels.

        Args:
            output_path (str): Path to save the visualization image.
        """
        # Extract node positions and attributes
        pos = {node: tuple(self.graph.nodes[node]['position'])
               for node in self.graph.nodes}
        node_colors = [
            # Échelle de gris avec transparence
            (luminance / 255, luminance / 255, luminance / 255, 0.6)
            for luminance in [self.graph.nodes[node]['avg_luminance'] for node in self.graph.nodes]
        ]
        node_sizes = [self.graph.nodes[node]['size']
                      * 0.5 for node in self.graph.nodes]

        # Set up a white background
        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('white')  # Set background color to white

        # Overlay the graph nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes
        )

        # Overlay the graph edges with black color
        edges = self.graph.edges()
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges,
            edge_color='black',  # Edges in black
            alpha=1.0,  # Full opacity
            width=1
        )

        # Add node labels (position and size) with black text and labels
        for node, (x, y) in pos.items():
            position = self.graph.nodes[node]['position']
            size = self.graph.nodes[node]['size']
            rounded_position = (round(position[0], 0), round(position[1], 0))
            label = f"Pos: {rounded_position}\nContours: {size}"

            plt.text(
                x, y, label,
                fontsize=6,
                color='black',  # Text color in black
                ha='center', va='center',
                bbox=dict(
                    facecolor='white',  # Label background color in white
                    alpha=0.9,  # Slightly opaque background
                    edgecolor='black',  # Border color in black
                    boxstyle='round,pad=0.3'
                )
            )

        # Save the graph visualization
        plt.axis('off')  # Remove axes
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()

    def visualize_graph_on_image(self, image_path, output_path=None):
        """
        Visualize the graph overlaid on the original image and save it.

        Args:
            image_path (str): Path to the original image.
            output_path (str): Path to save the overlaid image. If None, it will generate a path
                            based on the image_path.
        """
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Determine output_path based on image_path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"..\\..\\data\\output\\{base_name}_graph_overlay.png"

        # Extract node positions and attributes
        pos = {node: tuple(self.graph.nodes[node]['position'])
               for node in self.graph.nodes}
        node_colors = [
            # Échelle de gris avec transparence
            (luminance / 255, luminance / 255, luminance / 255, 0.6)
            for luminance in [self.graph.nodes[node]['avg_luminance'] for node in self.graph.nodes]
        ]
        node_sizes = [self.graph.nodes[node]['size']
                      * 0.5 for node in self.graph.nodes]

        # Plot the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Overlay the graph nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.gray
        )

        # Overlay the graph edges with white transparency
        edges = self.graph.edges()
        # White with 50% transparency
        edge_colors = [(1, 1, 1, 0.5)] * len(edges)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges,
            edge_color=edge_colors,
            alpha=0.5,  # Additional transparency
            width=1
        )

        # Add node labels (position and size)
        for node, (x, y) in pos.items():
            position = self.graph.nodes[node]['position']
            size = self.graph.nodes[node]['size']

            # Round position coordinates for clarity
            rounded_position = (round(position[0], 0), round(position[1], 0))
            label = f"Pos: {rounded_position}\nContours: {size}"

            plt.text(
                x, y, label,
                fontsize=6,
                color='white',  # Text color
                ha='center', va='center',
                bbox=dict(
                    facecolor=None,  # No background color
                    alpha=0,  # Transparent background
                    edgecolor=None,  # No border
                    boxstyle='round,pad=0.3'
                )
            )

        # Save the graph overlaid on the image
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_graph_like_overlay_but_without_image(self, image_path, output_path):
        """
        Draw the graph using the same coordinate space as the original image,
        but with a fully transparent background.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        height, width = image.shape[:2]

        pos = {node: tuple(self.graph.nodes[node]['position']) for node in self.graph.nodes}
        node_colors = [
            (luminance / 255, luminance / 255, luminance / 255, 0.6)
            for luminance in [self.graph.nodes[node]['avg_luminance'] for node in self.graph.nodes]
        ]
        node_sizes = [self.graph.nodes[node]['size'] * 0.5 for node in self.graph.nodes]

        # Use exact same size in inches and dpi to match the overlay
        dpi = 300
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis('off')
        ax.set_facecolor((0, 0, 0, 0))  # Transparent

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax
        )

        # Draw edges
        edge_colors = [(1, 1, 1, 0.5)] * len(self.graph.edges)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=self.graph.edges(),
            edge_color=edge_colors,
            alpha=0.5,
            width=1,
            ax=ax
        )

        # Draw labels
        for node, (x, y) in pos.items():
            size = self.graph.nodes[node]['size']
            rounded_position = (round(x, 0), round(y, 0))
            label = f"Pos: {rounded_position}\nContours: {size}"
            ax.text(
                x, y, label,
                fontsize=6,
                color='white',
                ha='center', va='center',
                bbox=dict(
                    facecolor='black',
                    alpha=0.5,
                    edgecolor='white',
                    boxstyle='round,pad=0.3'
                )
            )

        # Save exact-size image with transparent background
        fig.patch.set_alpha(0.0)
        plt.savefig(output_path, transparent=True, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
