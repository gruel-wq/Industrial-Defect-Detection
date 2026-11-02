
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a sample metal surface
image = np.ones((300, 400, 3), dtype=np.uint8) * 200

# 2. Add a simulated scratch
cv2.line(image, (100, 100), (280, 180), (30, 30, 30), 4)

# 3. Convert the image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Apply thresholding
_, threshold = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY_INV
)

# 5. Find defect boundaries
contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 6. Count detected defects
defect_count = 0

# 7. Draw rectangles around possible defects
for contour in contours:

    area = cv2.contourArea(contour)

    if area > 10:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        defect_count += 1

# 8. Display the detection result
if defect_count > 0:
    print("POSSIBLE DEFECT DETECTED")
    print("Number of detected regions:", defect_count)
else:
    print("NO DEFECT DETECTED")

# 9. Display all three images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(
    np.ones((300, 400, 3), dtype=np.uint8) * 200,
    cv2.COLOR_BGR2RGB
))
plt.title("Reference Surface")

plt.subplot(1, 3, 2)
plt.imshow(threshold, cmap="gray")
plt.title("Detected Dark Regions")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Defect Detection Result")

plt.tight_layout()
plt.show()
