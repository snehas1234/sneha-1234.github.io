import cv2

image_path = r'C:\Users\sneha\Desktop\animal.jpeg'
img = cv2.imread(image_path)
resize = cv2.resize(img, (650, 500))

def nothing(x):
    pass

cv2.namedWindow('Trackbar')
cv2.createTrackbar('Threshold1', 'Trackbar', 50, 255, nothing)
cv2.createTrackbar('Threshold2', 'Trackbar', 150, 255, nothing)

while True:
    thresh1 = cv2.getTrackbarPos('Threshold1', 'Trackbar')
    thresh2 = cv2.getTrackbarPos('Threshold2', 'Trackbar')

    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, thresh1, thresh2)

    cv2.imshow('Original Image', resize)
    cv2.imshow('Canny Edge Detection', canny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
