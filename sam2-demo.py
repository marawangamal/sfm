import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import show_points, show_masks

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model = build_sam2(model_cfg, checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


IMG_PATH = "images/ball.jpg"
INP_POINT = torch.tensor([[620, 720]])


def plot_image_with_point(image_path, input_point):

    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    input_label = torch.tensor([1])
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis("on")
    plt.show()


if __name__ == "__main__":
    # plot_image_with_point(IMG_PATH, INP_POINT)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = Image.open(IMG_PATH)
        predictor.set_image(image)

        input_label = torch.tensor([1])
        masks, scores, logits = predictor.predict(
            point_coords=INP_POINT,
            point_labels=input_label,
            multimask_output=True,
        )
        show_masks(
            image,
            masks,
            scores,
            point_coords=INP_POINT,
            input_labels=input_label,
            borders=True,
        )
