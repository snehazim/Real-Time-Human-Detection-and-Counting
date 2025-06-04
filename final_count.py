import numpy as np
import cv2

# Initialize
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cap = cv2.VideoCapture(0)  # Use webcam
cin = 0
cout = 0
pre = 0
prei = 800

# Use a supported codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Video_output.mp4', fourcc, 2, (680, 720), True)

# Main loop
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = img[80:, 100:]
    height, width, _ = img.shape
    img = cv2.medianBlur(img, 5)

    dilation = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(dilation, kernel, iterations=6)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 27])
    upper = np.array([200, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Thresholding
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Fix: use only 2 outputs (OpenCV 4+)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw center line
    img = cv2.line(img, (width // 2, 0), (width // 2, 600), (0, 0, 255), 4)

    for i in range(1, len(contours)):
        area = cv2.contourArea(contours[i])
        if 10000 < area < 25000:
            M = cv2.moments(contours[i])
            if M['m00'] == 0:
                continue  # Avoid division by zero

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(contours[i])

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crossing logic
            cur = cx

            def iscrossin(prei, cur): return prei < width / \
                2 and cur > width / 2

            def iscrossout(prei, cur): return prei > width / \
                2 and cur < width / 2

            if iscrossin(prei, cur):
                if abs(prei - cur) < 60:
                    cout += 1
            elif iscrossout(prei, cur):
                if abs(pre - cur) < 60:
                    cin += 1

            pre = cur
            prei = cur

    # Display counts
    cv2.putText(img, f"IN: {cin}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"OUT: {cout}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show and write frame
    cv2.imshow('image', img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
