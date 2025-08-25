import cv2
import os

def print_image_information(image):
    img = cv2.imread(image)

    height, width, channels = img.shape
    size = img.size
    data_type = img.dtype

    print("=== Image Information ===")
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size (number of elements): {size}")
    print(f"Data type: {data_type}")


def save_camera_information(output_file):
    cap = cv2.VideoCapture(0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if fps == 0:
        fps = "Unknown"

    with open(output_file, "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")

    print("\n=== Camera Information ===")
    print(f"fps: {fps}")
    print(f"width: {int(width)}")
    print(f"height: {int(height)}")

    cap.release()


if __name__ == "__main__":
    print_image_information("lena.png")

    output_path = os.path.join("solutions", "camera_outputs.txt")
    os.makedirs("solutions", exist_ok=True)  # make sure folder exists
    save_camera_information(output_path)
