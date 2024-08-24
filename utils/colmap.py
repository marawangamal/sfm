from typing import Tuple

import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pycolmap


def colmap_pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    """Decodes a pair_id into individual image IDs.

    Args:
        pair_id (int): Encoded pair ID combining two image IDs.

    Returns:
        Tuple[int, int]: A tuple containing the first and second image IDs.
    """
    image_id2 = pair_id % 2147483647
    image_id1 = (
        pair_id - image_id2
    ) // 2147483647  # Use integer division to get the correct image_id1
    return int(image_id1), int(image_id2)


def visualize_keypoints(image_path: pathlib.Path, keypoints: np.ndarray) -> None:
    """Visualizes keypoints on the given image.

    Args:
        image_path (pathlib.Path): Path to the image file.
        keypoints (np.ndarray): Array of keypoints, where each keypoint is a tuple (x, y).

    Returns:
        None: Displays the image with keypoints overlaid.
    """
    image = cv2.imread(str(image_path))

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Draw keypoints on the image
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])

        # Ensure keypoints are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(
                image, (x, y), 5, (0, 255, 0), thickness=2
            )  # Larger radius and thickness

    # Display the image with keypoints
    plt.figure(figsize=(10, 10))  # Increase figure size for better visibility
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axes
    plt.show()


def visualize_matches(
    image_path1: pathlib.Path,
    keypoints1: np.ndarray,
    image_path2: pathlib.Path,
    keypoints2: np.ndarray,
    matches: np.ndarray,
) -> None:
    """Visualizes the matches between two images by drawing lines between matching keypoints.

    Args:
        image_path1 (pathlib.Path): Path to the first image.
        keypoints1 (np.ndarray): Array of keypoints for the first image.
        image_path2 (pathlib.Path): Path to the second image.
        keypoints2 (np.ndarray): Array of keypoints for the second image.
        matches (np.ndarray): Array of matched keypoints indices.

    Returns:
        None: Displays the combined image with lines connecting matching keypoints.
    """
    image1 = cv2.imread(str(image_path1))
    image2 = cv2.imread(str(image_path2))

    if image1 is None or image2 is None:
        print(f"[ERROR] Unable to load images at {image_path1} and {image_path2}")
        return

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    output_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    output_image[:height1, :width1] = image1
    output_image[:height2, width1:] = image2

    # Draw lines between matching keypoints
    for match in matches:
        pt1 = (int(keypoints1[match[0]][0]), int(keypoints1[match[0]][1]))
        pt2 = (int(keypoints2[match[1]][0]) + width1, int(keypoints2[match[1]][1]))
        cv2.line(output_image, pt1, pt2, (0, 255, 0), 2)  # Increased line thickness

    # Display the image with matches
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axes
    plt.show()
    print(f"Visualized {len(matches)} matches")


def visualize_reconstruction(reconstruction: pycolmap.Reconstruction) -> None:
    """Visualizes the 3D points from the reconstruction.

    Args:
        reconstruction (pycolmap.Reconstruction): The 3D reconstruction to visualize.

    Returns:
        None: Displays a 3D plot of the reconstructed points.
    """
    points3D = np.array([point.xyz for point in reconstruction.points3D.values()])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Reconstruction")
    plt.show()
    print(f"Visualized {len(points3D)} 3D points")


def visualize_dense_reconstruction(mvs_path: pathlib.Path) -> None:
    """Visualizes the dense 3D reconstruction.

    Args:
        mvs_path (pathlib.Path): Path to the directory containing the dense reconstruction results.

    Returns:
        None: Displays a 3D visualization of the dense point cloud or mesh.
    """
    # Load the point cloud or mesh from the dense reconstruction
    dense_cloud_path = (
        mvs_path / "fused.ply"
    )  # Assuming the output is saved as fused.ply
    if dense_cloud_path.exists():
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(str(dense_cloud_path))

        # Visualize the point cloud
        o3d.visualization.draw_geometries(
            [pcd], window_name="Dense Reconstruction", width=800, height=600
        )
    else:
        print(
            f"[ERROR] Dense cloud not found at {dense_cloud_path}. Ensure that the reconstruction completed successfully."
        )
