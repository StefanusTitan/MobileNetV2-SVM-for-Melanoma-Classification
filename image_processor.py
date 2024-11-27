import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ImageProcessor:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def extract_roi(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        roi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return roi[y:y+h, x:x+w]

    def resize_and_pad(self, image):
        old_size = image.shape[:2]
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        resized = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = self.target_size - new_size[1]
        delta_h = self.target_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    def normalize(self, image):
        return preprocess_input(image.astype(np.float32))