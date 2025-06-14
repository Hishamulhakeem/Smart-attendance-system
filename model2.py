import os
import cv2
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# from google.colab.patches import cv2_imshow

IMG_SIZE = 160          
THRESHOLD = 0.50 

def buildEmbeddding():
    baseM = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                             include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(baseM.output)
    
    x = Dense(128, activation=None, name='embedding')(x)
    model = Model(inputs=baseM.input, outputs=x)
    model.save("face_embedding_model.keras")
    return model

embeddingModel = buildEmbeddding()
#embeddingModel.save("face_embedding_model.keras")
#embeddingModel.summary()

#Pre trained model
proto ="deploy.prototxt.txt"
caffe ="res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(proto, caffe)

def FaceDetection(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = preprocess_input(img) # Using  MobileNetV2's input for normalization.
    return img

def prepareEmbeddings(dataset_dir):
    embeddings = []
    labels = []
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            boxes = FaceDetection(img)
            if not boxes:
                print(f"[Warning] No face found in {img_path}")
                continue
            x1, y1, x2, y2 = boxes[0]
            face = img[y1:y2, x1:x2]
            face = preprocess(face)
            emb = embeddingModel.predict(np.expand_dims(face, axis=0))[0]
            emb_norm = emb / np.linalg.norm(emb)
            embeddings.append(emb_norm)
            labels.append(person_name)
    return np.array(embeddings), labels

def recognizeFace(image_path, known_embeddings, known_labels):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Unable to load image at {image_path}")
        return {}, []
    
    image = cv2.resize(image, None, fx=1.5, fy=1.5)
    boxes = FaceDetection(image)
    present = set()
    detection_details = []

    for (x1, y1, x2, y2) in boxes:
        face = image[y1:y2, x1:x2]
        face_proc = preprocess(face)
        emb = embeddingModel.predict(np.expand_dims(face_proc, axis=0))[0]
        emb_norm = emb / np.linalg.norm(emb)

        sims = cosine_similarity([emb_norm], known_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        print(f"Comparing face to knowns... Best match: {known_labels[best_idx]} | Score: {best_score:.2f}")

        if best_score > THRESHOLD:
            name = known_labels[best_idx]
        else:
            name = "Unknown"

        detection_details.append({
            "name": name,
            "score": float(best_score),
            "box": (x1, y1, x2, y2)
        })

        if name != "Unknown":
            present.add(name)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({best_score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    max_width = 1000
    scale = max_width / image.shape[1] if image.shape[1] > max_width else 1.0
    output_img = cv2.resize(image, None, fx=scale, fy=scale)

    cv2.imshow("Face Recognition Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    all_students = sorted(set(known_labels))
    attendance = {name: ("Present" if name in present else "Absent") 
                  for name in all_students}
    return attendance, detection_details

def save_attendance(attendance_dict):
    with open("attendance_report.json", "w") as jf:
        json.dump(attendance_dict, jf, indent=2)
    
    df = pd.DataFrame(list(attendance_dict.items()), columns=["Name", "Status"])
    df.to_csv("attendance_report.csv", index=False)
    print("[Info] Attendance report saved to attendance_report.json and attendance_report.csv.")
    
if __name__ == "__main__":
    
    dataset_folder = "datasets"      
    test_img = "inputimages/cctv2.jpg"           
    print("[Info] Preparing known face embeddings...")
    known_embs, known_labels = prepareEmbeddings(dataset_folder)
    
    print("[Info] Recognizing faces in classroom image...")
    attendance, detection_details = recognizeFace(test_img, known_embs, known_labels)

    print("\n--- Attendance Report ---")
    for name, status in attendance.items():
        print(f"{name}: {status}")
        

    save_attendance(attendance)