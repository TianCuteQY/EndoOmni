import cv2
import numpy as np


def mask_black(img):
    # convert to grayscale
    try:
        if img is None:
            raise ValueError("Image not loaded correctly")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
    # Handle or log the error appropriately

   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert gray image
    gray = 255 - gray

    # threshold
    thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]

    # invert thresh
    thresh = 255 - thresh

    return thresh > 0

    # get contours (presumably just one around the nonzero pixels)
    contours_output = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_output[0] if len(contours_output) == 2 else contours_output[1]

    # find the largest contour, which we assume to be contours[0] as per your requirement
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Create a mask where white is what we want, black otherwise
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

        # Convert mask to boolean where true is the area of interest
        mask = mask == 255

        return mask
    else:
        return np.zeros_like(gray)

if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt
    import os

    path = "/mnt/samba_share/ROBUST-MIS/Raw Data/Rectal Resection/3/frames/"
    image_path = path + os.listdir(path)[0]
    print(image_path)
    image = imageio.imread(image_path)

    image = np.array(image)

    plt.imshow(image)
    plt.title('Binary Mask')
    plt.show()

    mask = mask_black(image)

    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask')
    plt.show()

