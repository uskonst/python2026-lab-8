import cv2        

IMAGE_PATH = "apple.jpeg"    
OUTPUT_PATH = "apple_hsv.jpg"


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Файл не найден: {path}")
    return image


def convert_bgr_to_hsv(image) :
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


if __name__ == "__main__":
    bgr_image = load_image(IMAGE_PATH)
    hsv_image = convert_bgr_to_hsv(bgr_image)