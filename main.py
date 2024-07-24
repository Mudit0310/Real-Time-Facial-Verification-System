import cv2
import os
from deepface import DeepFace

folder_path = 'F:/Mudit/Pyhton II/Face/database/'

reference_images = []
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(folder_path, filename)
        ref_img = cv2.imread(img_path)
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        ref_faces = face_classifier.detectMultiScale(ref_gray, 1.1, 5, minSize=(40, 40))

        (x, y, w, h) = ref_faces[0]
        ref_face_img = ref_img[y:y + h, x:x + w]
        reference_images.append((ref_face_img, ref_img))

video = cv2.VideoCapture(0)

while True:
    result, frame = video.read()
    if not result:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    face_match = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face_img = frame[y:y + h, x:x + w]

        for ref_face_img, ref_img in reference_images:
            try:
                result = DeepFace.verify(face_img, ref_face_img)
                if result["verified"]:
                    face_match = True
                    matched_img = ref_img
                    break
            except Exception as e:
                print("Error:", e)

        if face_match:
            break

    if face_match:
        cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Matched Photo', matched_img)
    else:
        cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()