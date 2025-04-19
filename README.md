# SpamDetection

A machine learning-based web application for classifying SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.

---

## 🚀 Live Demo

Experience the application live at:  
👉 [https://ne5fxdynn4pmdqjfkzqste.streamlit.app](https://ne5fxdynn4pmdqjfkzqste.streamlit.app)

---

## 📌 Project Overview

This project leverages the **Multinomial Naive Bayes** classifier and **TF-IDF Vectorization** to process and classify SMS messages. The model is trained on the SMS Spam Collection dataset, achieving high accuracy in distinguishing between spam and non-spam messages.

---

## 🛠️ Features

- **Text Preprocessing**: Tokenization, stopword removal, and stemming.
- **Vectorization**: Conversion of text data into numerical format using TF-IDF.
- **Classification**: Utilization of a trained Naive Bayes model for prediction.
- **Web Interface**: User-friendly interface built with Streamlit for real-time predictions.

---

## 📂 Project Structure

```
SpamDetection/
├── app.py                # Main Streamlit application
├── model.pkl             # Trained classification model
├── vectorizer.pkl        # TF-IDF vectorizer
├── spam.csv              # Dataset used for training
├── nltk.txt              # NLTK data download script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DaivikM/SpamDetection.git
cd SpamDetection
```

### 2. Set Up a Virtual Environment

For Windows:

```bash
conda create -p venv python==3.9 -y
conda activate venv/
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Access the application at **http://localhost:8501**.

---

## 📄 Dataset

The model is trained on the **SMS Spam Collection** dataset, which contains a collection of SMS messages labeled as ham (non-spam) or spam. The dataset is publicly available and can be found in the repository.

---

## 🧪 Usage

1. Launch the application using the command:

   ```bash
   streamlit run app.py
   ```

2. Enter an SMS message in the provided text area.

3. Click on the **"Predict"** button to classify the message.

4. The application will display whether the message is **Spam** or **Not Spam**.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** – for creating the web interface
- **scikit-learn** – for machine learning algorithms
- **NLTK** – for natural language processing tasks
- **pickle** – for model serialization

---

## 📜 License

This project is licensed under the **MIT License**.

You can freely use, modify, and distribute the software with attribution, and without any warranty. See the [LICENSE](LICENSE) file for more details.

---

## 📞 Contact

For questions or support, feel free to reach out:
- Email: [dmohandm11@gmail.com](mailto:dmohandm11@gmail.com)
- GitHub: [DaivikM](https://github.com/DaivikM)

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your proposed changes.

---
