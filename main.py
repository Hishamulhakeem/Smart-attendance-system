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
from flask import Flask, request, jsonify, render_template, send_from_directory
import base64
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 4 MB max image size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


IMG_SIZE = 160
THRESHOLD = 0.76
embeddingModel = None
faceNet = None

def initialize_models():
    global embeddingModel, faceNet
    

    if not os.path.exists("face_embedding_model1.keras"):
        baseM = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                           include_top=False, weights='imagenet')
        x = GlobalAveragePooling2D()(baseM.output)
        x = Dense(128, activation=None, name='embedding')(x)
        embeddingModel = Model(inputs=baseM.input, outputs=x)
        embeddingModel.save("face_embedding_model1.keras")
    else:
        embeddingModel = tf.keras.models.load_model("face_embedding_model1.keras")
    
    proto = "deploy.prototxt.txt"
    caffe = "res10_300x300_ssd_iter_140000.caffemodel"
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

def preprocess_face(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

def prepare_known_embeddings():
    dataset_dir = "classroom_datasets"
    embeddings = []
    labels = []
    
    print(f"[DEBUG] Looking for dataset directory: {dataset_dir}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Directory exists: {os.path.exists(dataset_dir)}")
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory '{dataset_dir}' not found!")
        return np.array([]), []
    
    items = os.listdir(dataset_dir)
    print(f"[DEBUG] Items in dataset directory: {items}")
    
    person_count = 0
    total_images = 0
    
    for person_name in items:
        person_dir = os.path.join(dataset_dir, person_name)
        print(f"[DEBUG] Checking: {person_dir}")
        
        if not os.path.isdir(person_dir):
            print(f"[DEBUG] Skipping {person_name} - not a directory")
            continue
            
        person_count += 1
        person_images = 0
        
        try:
            person_files = os.listdir(person_dir)
            print(f"[DEBUG] Files in {person_name} folder: {person_files}")
        except Exception as e:
            print(f"[ERROR] Cannot read directory {person_dir}: {e}")
            continue
            
        for img_name in person_files:
            img_path = os.path.join(person_dir, img_name)
            print(f"[DEBUG] Processing image: {img_path}")
            
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"[DEBUG] Skipping {img_name} - not an image file")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not load image: {img_path}")
                continue
                
            print(f"[DEBUG] Image loaded successfully: {img.shape}")
            
            boxes = FaceDetection(img)
            print(f"[DEBUG] Detected {len(boxes)} faces in {img_name}")
            
            if not boxes:
                print(f"[WARNING] No face found in {img_path}")
                continue
                
            x1, y1, x2, y2 = boxes[0]
            print(f"[DEBUG] Face coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Invalid face coordinates in {img_path}")
                continue
                
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print(f"[WARNING] Empty face region in {img_path}")
                continue
                
            try:
                face = preprocess_face(face)
                emb = embeddingModel.predict(np.expand_dims(face, axis=0))[0]
                emb_norm = emb / np.linalg.norm(emb)
                embeddings.append(emb_norm)
                labels.append(person_name)
                person_images += 1
                total_images += 1
                print(f"[SUCCESS] Processed face for {person_name} from {img_name}")
            except Exception as e:
                print(f"[ERROR] Failed to process face from {img_path}: {e}")
                continue
        
        print(f"[INFO] Processed {person_images} images for {person_name}")
    
    print(f"[SUMMARY] Total persons: {person_count}, Total face embeddings: {len(embeddings)}")
    
    if len(embeddings) == 0:
        print("[ERROR] No face embeddings created! Check your dataset structure and image quality.")
    
    return np.array(embeddings), labels

def process_image_for_attendance(image_path):
    known_embeddings, known_labels = prepare_known_embeddings()
    
    if len(known_embeddings) == 0:
        return {"error": "No known faces found in classrooms_datasets folder"}, None
    
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Unable to load image"}, None
    
    original_image = image.copy()
    boxes = FaceDetection(image)
    present = set()
    detection_details = []

    for (x1, y1, x2, y2) in boxes:
        face = image[y1:y2, x1:x2]
        face_proc = preprocess_face(face)
        emb = embeddingModel.predict(np.expand_dims(face_proc, axis=0))[0]
        emb_norm = emb / np.linalg.norm(emb)

        sims = cosine_similarity([emb_norm], known_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score > THRESHOLD:
            name = known_labels[best_idx]
        else:
            name = "Unknown"

        detection_details.append({
            "name": name,
            "score": float(best_score),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

        if name != "Unknown":
            present.add(name)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({best_score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    all_students = sorted(set(known_labels))
    attendance = {name: ("Present" if name in present else "Absent") 
                  for name in all_students}

    _, buffer = cv2.imencode('.jpg', image)
    processed_img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return attendance, processed_img_b64


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            attendance, processed_image = process_image_for_attendance(filepath)
            
            if isinstance(attendance, dict) and "error" in attendance:
                return jsonify(attendance)
            
            with open("attendance_report.json", "w") as jf:
                json.dump(attendance, jf, indent=2)
            
            df = pd.DataFrame(list(attendance.items()), columns=["Name", "Status"])
            df.to_csv("attendance_report.csv", index=False)
            
            # ------------ New code for message if he is  absent 
            """
            import pywhatkit
            import time

            contact_directory = {
                'hisham': '9113022717',
                'sam': '9902366063',
                'karthick': '7411461709',
                'vinay': '7338610613',
                'sandesh': '9945025733',
                'prajwal': '6362105484'
                
            }

            for student, status in attendance.items():
                if status == "Absent":
                    phone = contact_directory.get(student.lower())
                    if phone:
                        message = f"Dear parent of {student}, your child was marked ABSENT today."
                        try:
                            pywhatkit.sendwhatmsg_instantly(
                                phone_no=f"+91{phone}",
                                message=message,
                                wait_time=20,     
                                tab_close=True     
                            )
                            print(f"[WHATSAPP SENT] Message to {student}'s parent at +91{phone}")
                            time.sleep(5)  
                        except Exception as e:
                            print(f"[ERROR] Could not send WhatsApp message to {student}: {e}")
                    else:
                        print(f"[NO CONTACT] Phone number not found for: {student}")
             """           
            return jsonify({
                'success': True,
                'attendance': attendance,
                'processed_image': processed_image
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'})
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)



@app.route('/get_students')
def get_students():
    dataset_dir = "classroom_datasets"
    students = []
    if os.path.exists(dataset_dir):
        students = [name for name in os.listdir(dataset_dir) 
                   if os.path.isdir(os.path.join(dataset_dir, name))]
    return jsonify({'students': sorted(students)})

@app.route('/debug_dataset')
def debug_dataset():
    dataset_dir = "classroom_datasets"
    debug_info = {
        'current_directory': os.getcwd(),
        'dataset_exists': os.path.exists(dataset_dir),
        'dataset_path': os.path.abspath(dataset_dir),
        'folders': [],
        'total_images': 0
    }
    
    if os.path.exists(dataset_dir):
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                debug_info['folders'].append({
                    'name': item,
                    'path': item_path,
                    'image_count': len(images),
                    'images': images[:5]  # Show first 5 images
                })
                debug_info['total_images'] += len(images)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, port=5000)
