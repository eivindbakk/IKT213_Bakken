import cv2
import numpy as np
import os

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    return padded_image

def crop(image, x_0, x_1, y_0, y_1):
    height, width = image.shape[:2]
    cropped_image = image[y_0:height-y_1, x_0:width-x_1]
    return cropped_image

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def copy(image, emptyPictureArray):
    height, width = image.shape[:2]
    if len(image.shape) == 3:
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    emptyPictureArray[y, x, c] = image[y, x, c]
    else:
        for y in range(height):
            for x in range(width):
                emptyPictureArray[y, x] = image[y, x]
    return emptyPictureArray

def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            for c in range(3):
                new_value = int(image[y, x, c]) + hue
                emptyPictureArray[y, x, c] = np.clip(new_value, 0, 255)
    return emptyPictureArray

def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)
    return smoothed_image

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

if __name__ == "__main__":
    original_image = cv2.imread("lena.png")
    if original_image is None:
        print("Error: Could not load lena.png. Please ensure the file exists in the current directory.")
        exit(1)
    print("Processing images...")
    padded = padding(original_image, 100)
    cv2.imwrite(os.path.join(output_dir, "padded_lena.png"), padded)
    print("✓ Padding completed")
    cropped = crop(original_image, 80, 130, 80, 130)
    cv2.imwrite(os.path.join(output_dir, "cropped_lena.png"), cropped)
    print("✓ Cropping completed")
    resized = resize(original_image, 200, 200)
    cv2.imwrite(os.path.join(output_dir, "resized_lena.png"), resized)
    print("✓ Resizing completed")
    height, width, channels = original_image.shape
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
    copied = copy(original_image, emptyPictureArray)
    cv2.imwrite(os.path.join(output_dir, "copied_lena.png"), copied)
    print("✓ Manual copy completed")
    gray = grayscale(original_image)
    cv2.imwrite(os.path.join(output_dir, "grayscale_lena.png"), gray)
    print("✓ Grayscale conversion completed")
    hsv_img = hsv(original_image)
    cv2.imwrite(os.path.join(output_dir, "hsv_lena.png"), hsv_img)
    print("✓ HSV conversion completed")
    emptyPictureArray2 = np.zeros((height, width, channels), dtype=np.uint8)
    hue_shift = hue_shifted(original_image, emptyPictureArray2, 50)
    cv2.imwrite(os.path.join(output_dir, "hue_shifted_lena.png"), hue_shift)
    print("✓ Hue shifting completed")
    smoothed = smoothing(original_image)
    cv2.imwrite(os.path.join(output_dir, "smoothed_lena.png"), smoothed)
    print("✓ Smoothing completed")
    rotated_90 = rotation(original_image, 90)
    cv2.imwrite(os.path.join(output_dir, "rotated_90_lena.png"), rotated_90)
    print("✓ 90° rotation completed")
    rotated_180 = rotation(original_image, 180)
    cv2.imwrite(os.path.join(output_dir, "rotated_180_lena.png"), rotated_180)
    print("✓ 180° rotation completed")
    print("\nAll image processing tasks completed successfully!")
    print(f"Output images saved in '{output_dir}' directory")
    print("\nOutput image dimensions:")
    print(f"  Original: {original_image.shape}")
    print(f"  Padded: {padded.shape}")
    print(f"  Cropped: {cropped.shape}")
    print(f"  Resized: {resized.shape}")
    print(f"  Grayscale: {gray.shape}")