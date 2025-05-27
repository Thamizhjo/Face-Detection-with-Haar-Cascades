# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

### PROGRAM
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

model = cv2.imread('WhatsApp Image 2024-01-09 at 21.26.03_7aea8f7c.jpg',0)
withglass = cv2.imread('WITHGLASS.png',0)
group = cv2.imread('GROUP.png',0)

plt.imshow(model,cmap='gray')
plt.show()
```
![Screenshot 2025-05-27 170611](https://github.com/user-attachments/assets/1deecee0-4ce9-4a8b-b3db-fa72f3d4289a)


```
plt.imshow(withglass,cmap='gray')
plt.show()
```
![Screenshot 2025-05-27 170617](https://github.com/user-attachments/assets/60ecc2fd-66f6-4ea8-9a76-09ff5578c5f8)


```
plt.imshow(group,cmap='gray')
plt.show()
```
![Screenshot 2025-05-27 170621](https://github.com/user-attachments/assets/16445576-846d-432b-84cc-8e78cbc3591c)


```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2) 
        
    return face_img
result = detect_face(withglass)


plt.imshow(result,cmap='gray')
plt.show()

```
![Screenshot 2025-05-27 170628](https://github.com/user-attachments/assets/41404dc4-f331-4bee-aa5f-930979688cdb)


```
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```
![445521496-acbc2fc8-5a65-4f58-844f-9d61f41ff4ac](https://github.com/user-attachments/assets/0117c1c0-0c9c-4389-9d68-b69be889c40a)


```


def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 4) 
        
    return face_img
    

# Doesn't detect the side face.
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```

![445521209-422acc22-215e-4578-9714-e88bc9cbf769](https://github.com/user-attachments/assets/da1722ce-1364-484c-8ebe-9d17eeee7ae4)


```

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 4) 
        
    return face_img
    

result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
```

![Screenshot 2025-05-27 170654](https://github.com/user-attachments/assets/124538e8-0243-46d5-9f23-f3f5d884259f)



```

eyes = eye_cascade.detectMultiScale(withglass)
# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```

![445522043-c2fa37fc-8344-487a-a8dc-752b9afd92a7](https://github.com/user-attachments/assets/99686b16-6f17-4d49-82e6-a2f496ac1b24)


```


cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()

```

![Screenshot 2025-05-27 170701](https://github.com/user-attachments/assets/a94e9ef6-2b7f-4384-ace1-d6e34f1a77f3)

