import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./kaggle/input/helmet-detection/images/BikesHelmets0.png")

dh, dw, _ = img.shape

fl = open("./kaggle/working/labels/BikesHelmets0.txt", 'r')
data = fl.readlines()
fl.close()

for dt in data:
    _, x, y, w, h = map(float, dt.split(' '))
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)

plt.imshow(img)
plt.show()



