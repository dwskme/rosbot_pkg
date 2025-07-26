import cv2
import numpy as np


def postprocess_mask(mask, apply_gaussian=True, kernel_size=5):
    if apply_gaussian:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def hough_lines(mask):
    h, w = mask.shape
    roi = mask[h // 2 :, :]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=5
    )
    line_img = np.zeros_like(mask)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_img, (x1, y1 + h // 2), (x2, y2 + h // 2), 255, 2)
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angles:
        return line_img, float(np.median(angles))
    # Fallback: centroid-based
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        fallback = np.degrees(np.arctan2((w // 2 - cx), cy))
    else:
        fallback = 0.0
    return line_img, fallback


def process_prediction(pred_tensor):
    mask = pred_tensor.squeeze().cpu().numpy()
    binary = (mask > 0.5).astype(np.uint8) * 255
    clean = postprocess_mask(binary)
    lines, angle = hough_lines(clean)
    return clean, lines, angle
