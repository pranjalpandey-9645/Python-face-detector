import cv2
import os

# 1. CASCADE PATH: Uses OpenCV's built-in reference to find the XML inside your venv
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
detect = cv2.CascadeClassifier(cascade_path)

# 2. IMAGE PATH NAVIGATION:
# os.path.abspath(__file__) gets: .../python-face-detector/src/app.py
# First dirname gets the 'src' folder: .../python-face-detector/src
# Second dirname gets the project root folder: .../python-face-detector
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Now target 'larry.jpg' perfectly inside the 'assets' folder
image_path = os.path.join(PROJECT_ROOT, "assets", "larry.jpg")

# 3. READ THE IMAGE: Standard method for reading static image files
img = cv2.imread(image_path)

# Safety check to catch any typos in file names or folder structures
if img is None:
    print(f"Error: Could not find or open the image at: {image_path}")
    print("Please double-check that 'larry.jpg' is spelled correctly inside your 'assets' folder.")
    exit()

# Convert the image to grayscale for face detection processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run the detection algorithm
faces = detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw cyan rectangles around any detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Display the final output window
cv2.imshow("Larry Image", img)

# Pause execution until a key is pressed, then close windows cleanly
cv2.waitKey(0)
cv2.destroyAllWindows()