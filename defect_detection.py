import cv2
import matplotlib.pyplot as plt

# 1. Load the industrial image
image = cv2.imread("test_images/scratch.jpg")

if image is None:
    print("Image not found!")
    exit()

# 2. Remove the bottom measurement information
height, width = image.shape[:2]
image = image[:int(height * 0.90), :]

# 3. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Reduce small image noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 5. Identify very dark regions
_, threshold = cv2.threshold(
    blur, 70, 255, cv2.THRESH_BINARY_INV
)

# 6. Remove small noise
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (3, 3)
)

cleaned = cv2.morphologyEx(
    threshold,
    cv2.MORPH_OPEN,
    kernel
)

# 7. Find defect boundaries
contours, _ = cv2.findContours(
    cleaned,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 8. Prepare detection image
result = image.copy()
defect_count = 0

# 9. Detect suspicious regions
for contour in contours:

    area = cv2.contourArea(contour)

    if area > 100:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        defect_count += 1

cv2.imwrite("results/detected_scratch.jpg", result)

print("Detection image saved successfully!")

if defect_count > 0:
    print("Possible surface defects found!")
else:
    print("No suspicious regions found.")

# 11. Display images
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cleaned, cmap="gray")
plt.title("Noise Reduced Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Improved Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
