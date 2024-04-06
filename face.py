import cv2

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize object tracker (KCF tracker)
tracker = cv2.Tracker_create("KCF")

# Open video stream
video_capture = cv2.VideoCapture('input_video.mp4')

# Initialize bounding box
bbox = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if bbox is None:
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Select the first face as the object to track
        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)

    else:
        # Update the tracker
        success, bbox = tracker.update(frame)

        # Draw bounding box around the tracked object
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            bbox = None

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
