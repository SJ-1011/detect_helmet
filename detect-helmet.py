from ultralytics import YOLO
model = YOLO("C:/Users/user/runs/detect/train8/weights/best.pt")

from PIL import Image
import cv2
img = cv2.imread("input4.png")
prediction = model.predict(img)[0]

num = len(prediction)
print("Number of people detected: ",num)

prediction = prediction.plot(line_width=1)
prediction = prediction[:, :, ::-1]
prediction = Image.fromarray(prediction)
prediction.save("output4.png")

print()