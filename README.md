# suyunov_shohjahon
import cv2
import numpy as np


img = cv2.imread("image.jpg")

img = cv2.resize(img, (640, 480))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blur, 50, 150)

_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

path_mask = np.zeros_like(gray)
landing_zone = None

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 2000:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if landing_zone is None or area > landing_zone[0]:
            landing_zone = (area, x, y, w, h)


if landing_zone:
    _, x, y, w, h = landing_zone
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.putText(img, "Landing Zone", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


img[edges != 0] = [0, 0, 255]  


cv2.imshow("Original", img)
cv2.imshow("Edges (Obstacles)", edges)
cv2.imshow("Path Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
