
import cv2
import matplotlib.pyplot as plt

# Step 1: Open the real metal image
image = cv2.imread("test_images/scratch.jpg")

# Step 2: Check whether image is loaded
if image is None:
    print("Image not found! Check the file location.")
else:

    # Step 3: Convert image colours
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 4: Display the image
    plt.imshow(image)
    plt.title("Real Industrial Metal Image")
    plt.axis("off")
    plt.show()

    print("Image loaded successfully!")
