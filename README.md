# 🎓 Smart Attendance System Using Face Recognition

This project automates the process of student attendance using classroom images captured from CCTV or any high-resolution camera. The system detects and recognizes faces using deep learning and marks attendance in both CSV and JSON formats. If a student is absent, the system can optionally send a WhatsApp message to their parent.

---

## 🔍 Key Features

- 🔎 **Face Recognition**: Powered by MobileNetV2 with 128D embedding generation.
- 🧠 **Face Detection**: Based on OpenCV’s SSD + ResNet model.
- 💾 **Attendance Reports**: Automatically saved as `.csv` and `.json` files.
- 📩 **Parent Notification**: WhatsApp messages sent to parents of absentees.
- 🖼️ **Visual Feedback**: Bounding boxes and names with matching score drawn on detected faces.
- 💻 **Interactive Web Interface**: Upload images and view results instantly.

---

## 📥 Image Requirements

- 📸 **Resolution**: Minimum 480p if good lighting available or else 720p must  
- 🔄 **Max File Size**: Must be **under 4 MB**  
- 👥 **Student Capacity**: Up to **35–40 students** per image  
- 🔦 **Lighting**: Good lighting improves recognition accuracy  
- 🎥 **Camera Angle**:
  - If placed in the **corner**, captures up to ~**50° field of view** effectively.
  - **Center** placement is ideal for covering the full class.

---

## 🗂️ Dataset Info

- Each student has ~15 original images.
- With augmentation, each has **40–50 images total**.
- All images are organized in folders named after each student.

---

## 🧪 How It Works

1. 📤 Upload a classroom image.
2. 🧠 Faces are detected and embedded using a trained deep learning model.
3. 🧾 Attendance is computed by matching faces to the known dataset.
4. 📸 A result image is generated and saved.
5. 📝 Reports are saved as `attendance_report.csv` and `attendance_report.json`.
6. 📲 WhatsApp notifications are sent to parents of absentees (optional).

---

## 💻 Example Demo

### 🖼️ Input Image:
![Input](classroom_test2.jpg)

### 🧾 Output Image:
![Output](redme.jpg)

### 🌐 Web Interface Preview:
![UI](71e8cbf9-bd62-41b6-a336-34518c1c33aa.png)

---

## 👨‍🎓 Example Students

The system has been tested on the following students for demo purposes:

- Anya Taylor Joy  
- Fiennes Tiffin  
- Jenna Ortega  
- McLaughlin  
- Murnal  
- SRK  
- Tom Cruise  
- Tom Holland  
- Zendaya  

---

## ▶️ Getting Started

### ✅ Install Required Libraries

```bash
pip install opencv-python tensorflow flask pandas numpy scikit-learn pillow pywhatkit
