
# 🧠 Handwritten Digit Recognizer with Gradio

This project provides a simple web interface using **Gradio** to recognize handwritten digits (**0–9**).  
It uses a **pre-trained Convolutional Neural Network (CNN)** model built with **TensorFlow/Keras**.

---

## 🚀 Live Demo

Try the live version of this app hosted on **Hugging Face Spaces**:

👉 [**Handwritten Digit Recognizer on Hugging Face**](https://huggingface.co/spaces/venmugilrajan/DIGIT_RECOGNIZER_CNN)

---

## 📁 Project Structure

```

DIGIT_RECOGNIZER_CNN/
│
├── gradio_app.py          # Main Gradio app script
├── digit_cnn.keras        # Pre-trained CNN model file
├── README.md              # Project documentation
│
├── (Optional)
│   ├── code.ipynb         # Jupyter notebook used for training
│   └── app.py             # Script version of the notebook

````

---

## ⚙️ Requirements

Make sure you have **Python 3.7+** installed.

Install the required libraries using:

```bash
pip install gradio tensorflow numpy pillow
````

> 💡 If you want GPU acceleration, install a compatible TensorFlow GPU version.

---

## 🧩 Model Details

The model is a **Convolutional Neural Network (CNN)** trained on a handwritten digit dataset (similar to **MNIST**).
It includes:

* **Conv2D layers** → Extract spatial features from the image
* **MaxPooling2D layers** → Reduce spatial dimensions
* **Dropout layers** → Prevent overfitting
* **Dense layers** → Perform classification
* **Softmax activation** → Output probabilities for each digit (0–9)

The trained model is saved in the **Keras format (`.keras`)** and loaded in the Gradio app for prediction.

---

## 🧰 Setup and Usage

### 1️⃣ Clone or Download the Project

```bash
git clone https://github.com/venmugilrajan/DIGIT_RECOGNIZER_CNN.git
cd DIGIT_RECOGNIZER_CNN
```

### 2️⃣ Place the Model File

Ensure the **`digit_cnn.keras`** file is in the same directory as `gradio_app.py`.

### 3️⃣ Run the Application

```bash
python gradio_app.py
```

### 4️⃣ Access the Interface

After running the script, open the local URL shown in your terminal:

```
http://127.0.0.1:7860
```

---

## 🖌️ How to Use

1. Draw a single digit (**0–9**) in the sketchpad area.
2. Click **"Submit"**.
3. View the **top 3 predicted digits** and their **confidence scores**.

---

## 🧾 Example Output

| Drawn Digit | Predicted Digit | Confidence |
| ----------- | --------------- | ---------- |
| ✏️ 3        | ✅ 3             | 99.2%      |
| ✏️ 7        | ✅ 7             | 97.8%      |

---

## 💡 Notes

* Large files (like `train.csv`) should use **Git Large File Storage (Git LFS)** if needed.
  Learn more here: [https://git-lfs.github.com](https://git-lfs.github.com)
* Ensure your model file is **below 100MB** for GitHub compatibility.

---

## 👨‍💻 Author

**Venmugil Rajan S**
📧 [venmugilrajan@gmail.com](mailto:venmugilrajan@gmail.com)
🌐 [Hugging Face Profile](https://huggingface.co/venmugilrajan)

---

## 🏷️ License

This project is open-source and available under the **MIT License**.

---

⭐ **If you found this project helpful, please give it a star on GitHub!** ⭐

```

---

Would you like me to include a **preview image (screenshot of your Gradio app)** section too?  
It looks great at the top of a GitHub README — I can show you how to add that next.
```
