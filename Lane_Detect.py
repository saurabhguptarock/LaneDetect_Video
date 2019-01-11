import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, line):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


cap = cv2.VideoCapture('road.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    lane_image = np.copy(frame)
    image = canny(lane_image)
    image = region_of_interest(image)

    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    image = display_lines(lane_image, image)
    image = cv2.addWeighted(lane_image, 0.8, image, 1, 1)
    cv2.imshow('result', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
