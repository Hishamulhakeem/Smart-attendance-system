# 🎓 Smart Attendance System Using Face Recognition

This is a smart and efficient system to automatically mark student attendance by analyzing a classroom photo — typically captured from a CCTV camera or even a decent smartphone camera. The system identifies student faces, matches them with the dataset, and marks them as present or absent. It also generates detailed reports and can notify parents via WhatsApp if their child is absent.

---

## 🔍 Key Features

- 🔎 **Face Recognition**: Powered by MobileNetV2 with 128D embedding generation.
- 🧠 **Face Detection**: Based on OpenCV’s SSD + ResNet model.
- 💾 **Attendance Reports**: Automatically saved as `.csv` and `.json` files.
- 📩 **Parent Notification**: Sends WhatsApp alerts to parents of absent students.
- 🖼️ **Visual Feedback**: Highlights each recognized face with name and confidence score in BBox.
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

- Each student has ~10-15 original images.
- With augmentation, each has **40–50 images total**.
- All images are organized in folders named after each student.

---

## 🧪 How It Works

1. Upload a photo of the classroom.
2. The system detects faces and extracts embeddings.
3. It compares them with known student embeddings.
4. Students are marked **Present** or **Absent**.
5. Generates reports (`.csv`, `.json`) and an annotated image.
6. Optionally, it sends **WhatsApp messages** to parents of absentees.

---

## 👨‍🎓 Example Students with Demo

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


### 🖼️ Input Image:
![classroom_test2](https://github.com/user-attachments/assets/d3a24b48-273b-4944-938d-d78250472c40)


### 🧾 Output Image:
![redme](https://github.com/user-attachments/assets/aa342a37-157c-4e2c-9d85-87d1778758dc)

### 🌐 Web Interface Preview:
![image](https://github.com/user-attachments/assets/7664eb00-6325-410b-96fc-54f0129f7d0b)


---

## ▶️ Getting Started

### ✅ Install Required Libraries

```bash
pip install opencv-python tensorflow flask pandas numpy scikit-learn pillow pywhatkit
