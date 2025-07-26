import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    MotionBlur,
    Normalize,
    RandomBrightnessContrast,
    Resize,
)
from albumentations.pytorch import ToTensorV2


def apply_filters(img):
    # Equalize Y in YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def get_transforms():
    return Compose(
        [
            HorizontalFlip(p=0.5),
            MotionBlur(blur_limit=5, p=0.2),
            RandomBrightnessContrast(p=0.5),
            Resize(256, 256),
            Normalize(),
            ToTensorV2(),
        ]
    )
