# Real-Time Drosophila Detection App

Welcome to the **Real-Time Drosophila Detection App**! This application leverages the power of **Detectron2** and **Streamlit** to provide real-time detection of male and female Drosophila (fruit flies) using your webcam or captured images.



---

## 🚀 Features

- **Real-Time Detection:** Utilize your webcam to detect and visualize male and female Drosophila in real-time.
- **Image Capture:** Capture images and run detections on static photos.
- **Interactive Interface:** Easily switch between live video and image capture modes using a user-friendly sidebar.
- **Confidence Threshold Adjustment:** Dynamically adjust the confidence threshold to filter detections.
- **Visual Feedback:** View bounding boxes and labels for detected instances directly on the video feed or images.
- **Git Large File Storage (LFS):** Efficiently manage and store large model files.

---

## 📚 Table of Contents

- [🚀 Features](#-features)
- [🛠️ Installation](#️-installation)
- [⚙️ Usage](#️-usage)
- [📦 Dependencies](#-dependencies)
- [🤖 Model Information](#-model-information)
- [📄 License](#-license)
- [📫 Contact](#-contact)
- [🙏 Acknowledgments](#-acknowledgments)


---

## 🛠️ Installation

Follow these steps to set up and run the application on your local machine.

### **1. Clone the Repository**

```bash
git clone https://github.com/jayantrai12/important.git
cd important
```

### **2. Set Up Git Large File Storage (LFS)**

Ensure that Git LFS is installed to manage the large model file.

#### **Install Git LFS**

- **macOS:**

  ```bash
  brew install git-lfs
  ```

- **Ubuntu/Debian:**

  ```bash
  sudo apt-get install git-lfs
  ```

- **Windows:**

  Download and run the installer from [Git LFS Releases](https://github.com/git-lfs/git-lfs/releases).

#### **Initialize Git LFS**

```bash
git lfs install
```

### **3. Install Dependencies**

It's recommended to use a virtual environment to manage dependencies.

#### **Using `venv`:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **Install Required Packages:**

```bash
pip install -r requirements.txt
```

**Note:** If you don't have a `requirements.txt` file, create one with the following content:

```text
streamlit
opencv-python
numpy
torch
torchvision
torchaudio
av
pillow
detectron2
streamlit-webrtc
```

**Important:** Installing **Detectron2** may require additional steps based on your system and CUDA version. Refer to the [Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for detailed instructions.

### **4. Verify Model File**

Ensure that the model file `model_final_f10217.pkl` is present in the repository and tracked by Git LFS.

```bash
git lfs ls-files
```

You should see an entry like:

```
9a737e2903 * model_final_f10217.pkl
```

---

## ⚙️ Usage

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

### **App Interface**

1. **Select Input Method:**
   - **Live Video:** Use your webcam for real-time detection.
   - **Capture Image:** Take a snapshot and run detections on the captured image.

2. **Live Video Mode:**
   - Grant camera access when prompted by your browser.
   - Adjust the **Confidence Threshold** slider in the sidebar to filter detections based on confidence scores.
   - View real-time detections with bounding boxes and labels for male and female Drosophila.

3. **Capture Image Mode:**
   - Click on **"Take a picture"** to capture an image using your webcam.
   - The app will process the image and display detected instances with bounding boxes and labels.
   - View the number of detected instances and their respective classes and confidence scores.

### **Example Commands**

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

---

## 📦 Dependencies

- **Streamlit:** Interactive web framework for data apps.
- **OpenCV:** Computer vision library for image processing.
- **NumPy:** Fundamental package for scientific computing.
- **PyTorch:** Deep learning framework.
- **Detectron2:** Facebook AI Research's library for object detection.
- **Streamlit-WeRTC:** Real-time video and audio streaming in Streamlit.
- **Pillow:** Python Imaging Library for image processing.
- **Git LFS:** Manage large files in Git repositories.

Ensure all dependencies are installed correctly, especially **Detectron2**, which has specific installation requirements.

---

## 🤖 Model Information

### **Model Architecture**

- **Base Model:** Mask R-CNN with a ResNet-50-FPN backbone.
- **Configuration File:** `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`
- **Number of Classes:** 2 (`male` and `female` Drosophila)
- **Confidence Threshold:** Adjustable (default set to 0.3)

### **Model Training**

- **Dataset:** Custom-trained on a dataset containing `male` and `female` Drosophila.
- **Training Framework:** Detectron2 with configurations tailored for instance segmentation.
- **Weights File:** `model_final_f10217.pkl`

### **Usage**

The model is integrated into the Streamlit app to perform real-time detections on live video feeds or captured images, highlighting detected instances with bounding boxes and labels.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📫 Contact

**Jayantra Rai**

- **GitHub:** [@jayantrai12](https://github.com/jayantrai12)
- **Email:** jayantrai2810@gmail.com <!-- Replace with your actual email -->

Feel free to reach out for any questions, feedback, or collaboration opportunities!

---

## 🙏 Acknowledgments

- **[Detectron2](https://github.com/facebookresearch/detectron2):** Facebook AI Research's next-generation library for object detection and segmentation.
- **[Streamlit](https://streamlit.io/):** Fast, easy-to-use open-source framework for building custom ML tools.
- **[Streamlit-WeRTC](https://github.com/whitphx/streamlit-webrtc):** Enables real-time video and audio streaming in Streamlit apps.
- **Dr. Ishaan Gupta Sir:** Thank you for your guidance and support!

---

## 📈 Future Improvements

- **Enhanced Detection Algorithms:** Integrate more advanced models or fine-tune existing ones for improved accuracy.
- **Multiple Class Detection:** Expand the model to detect additional classes or categories.
- **User Authentication:** Implement user authentication for secure access to the app.
- **Deployment:** Deploy the app on cloud platforms for broader accessibility.
- **Performance Optimization:** Optimize the app for faster inference and reduced latency.

---

**Happy detecting! **

---
