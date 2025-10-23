import cv2
import numpy as np

def detect_yellow_ball():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range 
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius > 10:  # Filter by size
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # Show the output
        cv2.imshow('Yellow Tennis Ball Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_yellow_ball()

#draws circle and centre if radius>10
