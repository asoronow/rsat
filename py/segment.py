import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt


def sobel(image):
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
    gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)

    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    return combined


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax[1].imshow(img)


image = cv2.imread("./images/test_patch.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = sobel(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
sam = sam_model_registry["default"](checkpoint="./sam/checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)


fig, axes = plt.subplots(ncols=2, figsize=(8, 2.7))
ax = axes.ravel()
ax[0] = plt.subplot(1, 2, 1, adjustable="box")
ax[1] = plt.subplot(1, 2, 2, adjustable="box")
ax[0].imshow(image)
ax[0].set_title("Original")
ax[0].axis("off")
show_anns(masks)
ax[1].set_title("Mask")
ax[1].axis("off")
plt.show()
