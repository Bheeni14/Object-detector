import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


tracker = cv2.Tracker_create("KCF")


video_capture = cv2.VideoCapture('input_video.mp4')


bbox = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if bbox is None:
       
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

       
        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)

    else:
      
        success, bbox = tracker.update(frame)

       
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            bbox = None

  
    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
