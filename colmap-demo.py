import sqlite3
import pycolmap
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def visualize_keypoints(image_path, keypoints):
    image = cv2.imread(str(image_path))

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Increase circle size and thickness for better visibility
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])

        # Ensure keypoints are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(
                image, (x, y), 5, (0, 255, 0), thickness=2
            )  # Larger radius and thickness

    plt.figure(figsize=(10, 10))  # Increase figure size for better visibility
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axes
    plt.show()


def visualize_matches(image_path1, keypoints1, image_path2, keypoints2, matches):
    image1 = cv2.imread(str(image_path1))
    image2 = cv2.imread(str(image_path2))
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if image1 is None or image2 is None:
        print(
            f"[ERROR] Unable visualize matches - no images to load images at {image_path1} and {image_path2}"
        )
        return

    output_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    output_image[:height1, :width1] = image1
    output_image[:height2, width1:] = image2

    for match in matches:
        pt1 = (int(keypoints1[match[0]][0]), int(keypoints1[match[0]][1]))
        pt2 = (int(keypoints2[match[1]][0]) + width1, int(keypoints2[match[1]][1]))
        cv2.line(output_image, pt1, pt2, (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print(f"Visualized {len(matches)} matches")


def extract_features(database_path, image_dir, visualize=False):
    pycolmap.extract_features(database_path, image_dir)
    if visualize:
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()

        # Query images and their IDs
        cursor.execute("SELECT image_id, name FROM images")
        images = cursor.fetchall()

        # For each image, query keypoints
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


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (
        pair_id - image_id2
    ) // 2147483647  # Use integer division to get the correct image_id1
    return int(image_id1), int(image_id2)


def match_features(database_path, image_dir, visualize=False):
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
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)

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


def main():

    # (1) Find interest points in each image
    # (2) Find candidate correspondences (match descriptors for each interest point)
    # (3) Perform geometric verification of correspondences (RANSAC + fundamental matrix)
    # (4) Solve for 3D points and camera that minimize reprojection error.

    # Setup paths
    output_path = pathlib.Path("output")
    image_dir = pathlib.Path("images")
    database_path = output_path / "database.db"

    # (1) Extract features from images
    extract_features(database_path, image_dir, visualize=True)

    # (2) Match features between images
    match_features(database_path, image_dir, visualize=True)


if __name__ == "__main__":
    main()
