import cv2
import os
import numpy as np

def capture_images(face_cascade, camera_id=0, num_persons=5, num_images_per_person=100):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    for person_id in range(1, num_persons + 1):
        person_name = input(f"Enter name for person {person_id}: ")
        person_dir = f'data/{person_name}'
        os.makedirs(person_dir, exist_ok=True)

        print(f"Capturing images for {person_name}...")

        images_captured = 0
        while images_captured < num_images_per_person:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]

                cv2.imwrite(f'{person_dir}/{images_captured + 1}.jpg', roi_gray)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                images_captured += 1

                print(f"Captured image {images_captured} for {person_name}")

                if images_captured >= num_images_per_person:
                    break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def train_model(data_dir):
    known_faces = []
    known_labels = []
    label_map = {}

    label_id = 0
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)

        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name

        for image_file in os.listdir(person_dir):

            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                known_faces.append(image)
                known_labels.append(label_id)

        label_id += 1

    if not known_faces or not known_labels:
        print("Error: No training data found.")
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(known_faces, np.array(known_labels))

    return recognizer, label_map

def recognize_faces(face_cascade, recognizer, label_map, camera_id=0):

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Starting face recognition...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)
            name = label_map.get(label, "Unknown")
            print(f"Detected: {name} with confidence: {confidence:.2f}")

            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Error: Could not load face cascade.")

    else:
        # Step 1: Capture images
        capture_images(face_cascade)

        # Step 2: Train model
        recognizer, label_map = train_model('data')

        if recognizer:

            # Step 3: Recognize faces
            recognize_faces(face_cascade, recognizer, label_map)
