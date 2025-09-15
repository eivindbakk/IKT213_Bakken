import cv2
import numpy as np
import os

def create_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')

def sobel_edge_detection(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)

    sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

    sobel_combined = np.uint8(sobel_combined)

    _, sobel_enhanced = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

    cv2.imwrite('output/sobel_edges.jpg', sobel_enhanced)

    return sobel_enhanced

def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    cv2.imwrite('output/canny_edges.jpg', edges)

    return edges

def template_match(image, template):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template

    w, h = template_gray.shape[::-1]

    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9

    loc = np.where(result >= threshold)

    if len(image.shape) == 3:
        output = image.copy()
    else:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('output/template_match.jpg', output)

    return output

def resize(image, scale_factor, up_or_down):
    if up_or_down == "up":
        result = image
        for _ in range(scale_factor):
            result = cv2.pyrUp(result)
    elif up_or_down == "down":
        result = image
        for _ in range(scale_factor):
            result = cv2.pyrDown(result)
    else:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    cv2.imwrite(f'output/resized_{up_or_down}_{scale_factor}.jpg', result)

    return result

def main():
    create_output_folder()

    lambo = cv2.imread('lambo.png')
    if lambo is None:
        print("Error: Could not load lambo.png")
        return

    print("Performing Sobel edge detection...")
    sobel_result = sobel_edge_detection(lambo)

    print("Performing Canny edge detection...")
    canny_result = canny_edge_detection(lambo, 50, 50)

    print("Loading shapes for template matching...")
    shapes = cv2.imread('shapes.png')
    shapes_template = cv2.imread('shapes_template.jpg')

    if shapes is None or shapes_template is None:
        print("Error: Could not load shapes images")
    else:
        print("Performing template matching...")
        template_result = template_match(shapes, shapes_template)

    print("Performing resizing...")
    resize_up = resize(lambo, 2, "up")
    resize_down = resize(lambo, 2, "down")

    print("All operations completed successfully!")
    print("Results saved to the 'output' folder.")

if __name__ == "__main__":
    main()