import cv2
from mtcnn.mtcnn import MTCNN


class Models:
    OpenCV = 1
    MTCNN = 2


FACE_DETECTION_MODEL = Models.OpenCV


def capture_frames():
    face_cascade = detector = None
    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)  # 0 represents the video input by default

    # check if camera is open
    if not cap.isOpened():
        print("Error opening the camera")
        return

    # Load the Haar Cascade classifier to detect faces
    if FACE_DETECTION_MODEL == Models.OpenCV:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    elif FACE_DETECTION_MODEL == Models.MTCNN:
        detector = MTCNN()

    while True:
        # read frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading the frame")
            break

        faces = []
        if FACE_DETECTION_MODEL == Models.OpenCV:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        elif FACE_DETECTION_MODEL == Models.MTCNN:
            faces = detector.detect_faces(frame)
            faces = [face['box'] for face in faces]

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the frame in the window
        cv2.imshow('Webcam', frame)

        # Exit when pressing the key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


def main():
    capture_frames()


if __name__ == "__main__":
    main()
