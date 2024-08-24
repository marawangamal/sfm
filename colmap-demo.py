import sqlite3
import pycolmap
import pathlib
import numpy as np
from utils.colmap import (
    visualize_keypoints,
    visualize_matches,
    colmap_pair_id_to_image_ids,
)

# output_path: pathlib.Path = pathlib.Path("output")
# image_dir: pathlib.Path = pathlib.Path("images")

# mvs_path = output_path / "mvs"
# database_path = output_path / "database.db"

# out1 = pycolmap.extract_features(database_path, image_dir)
# out2 = pycolmap.match_exhaustive(database_path)
# maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
# # points = list(maps[0].points3D.values())
# maps[0].write(output_path)
# # dense reconstruction
# pycolmap.undistort_images(mvs_path, output_path, image_dir)

# reconstruction = pycolmap.Reconstruction("output")
# print(reconstruction.summary())

# for image_id, image in reconstruction.images.items():
#     print(image_id, image)

# for point3D_id, point3D in reconstruction.points3D.items():
#     print(point3D_id, point3D)

# for camera_id, camera in reconstruction.cameras.items():
#     print(camera_id, camera)
# reconstruction.write("output")


def extract_features(
    database_path: pathlib.Path, image_dir: pathlib.Path, visualize: bool = False
) -> None:
    """Extracts features from images using COLMAP and optionally visualizes the keypoints.

    Args:
        database_path (pathlib.Path): Path to the COLMAP database.
        image_dir (pathlib.Path): Path to the directory containing images.
        visualize (bool, optional): Whether to visualize keypoints of the first image. Defaults to False.

    Returns:
        None: Extracts features from images and stores them in the COLMAP database.
        If visualize is True, displays the keypoints for the first image.
    """
    pycolmap.extract_features(database_path, image_dir)
    if visualize:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # Query images and their IDs
        cursor.execute("SELECT image_id, name FROM images")
        images = cursor.fetchall()

        # For the first image only, query keypoints and visualize
        if images:
            image_id, image_name = images[0]
            cursor.execute(f"SELECT data FROM keypoints WHERE image_id={image_id}")
            keypoints_data = cursor.fetchone()

            if keypoints_data:
                keypoints = np.frombuffer(keypoints_data[0], dtype=np.float32).reshape(
                    -1, 6
                )[:, :2]
                image_path = image_dir / image_name
                visualize_keypoints(image_path, keypoints)
        conn.close()


def match_features(
    database_path: pathlib.Path, image_dir: pathlib.Path, visualize: bool = False
) -> None:
    """Matches features between images using COLMAP and optionally visualizes the matches.

    Args:
        database_path (pathlib.Path): Path to the COLMAP database.
        image_dir (pathlib.Path): Path to the directory containing images.
        visualize (bool, optional): Whether to visualize matches for the first pair of images. Defaults to False.

    Returns:
        None: Matches features between images and stores them in the COLMAP database.
        If visualize is True, displays the matches for the first image pair.
    """
    pycolmap.match_exhaustive(database_path)
    if visualize:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # Query the first match pair
        cursor.execute("SELECT pair_id, data FROM matches LIMIT 1")
        match_data = cursor.fetchone()

        if match_data:
            pair_id, matches_blob = match_data

            # Decode the pair_id to get the image IDs
            image_id1, image_id2 = colmap_pair_id_to_image_ids(pair_id)

            # Query image names based on image IDs
            cursor.execute("SELECT name FROM images WHERE image_id = ?", (image_id1,))
            image_name1 = cursor.fetchone()[0]
            cursor.execute("SELECT name FROM images WHERE image_id = ?", (image_id2,))
            image_name2 = cursor.fetchone()[0]

            # Retrieve keypoints for both images
            cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id1,))
            keypoints_data1 = cursor.fetchone()
            keypoints1 = np.frombuffer(keypoints_data1[0], dtype=np.float32).reshape(
                -1, 6
            )[:, :2]

            cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id2,))
            keypoints_data2 = cursor.fetchone()
            keypoints2 = np.frombuffer(keypoints_data2[0], dtype=np.float32).reshape(
                -1, 6
            )[:, :2]

            # Decode matches
            matches = np.frombuffer(matches_blob, dtype=np.uint32).reshape(-1, 2)

            # Visualize the matches
            image_path1 = image_dir / image_name1
            image_path2 = image_dir / image_name2
            visualize_matches(image_path1, keypoints1, image_path2, keypoints2, matches)

        conn.close()


def main() -> None:
    """Main function to extract features and match them using COLMAP.

    This function:
    1. Finds interest points in each image.
    2. Finds candidate correspondences (match descriptors for each interest point).
    3. Performs geometric verification of correspondences (RANSAC + fundamental matrix).
    4. Solves for 3D points and camera that minimize reprojection error.

    Args:
        None

    Returns:
        None
    """
    # Setup paths
    output_path = pathlib.Path("output")
    image_dir = pathlib.Path("images")
    database_path = output_path / "database.db"

    # (1) Extract features from images
    extract_features(database_path, image_dir, visualize=True)

    # (2) Match features between images
    match_features(database_path, image_dir, visualize=True)

    # (3) Perform geometric verification of correspondences (RANSAC + fundamental matrix)
    # (4) Solve for 3D points and camera that minimize reprojection error.


if __name__ == "__main__":
    main()
