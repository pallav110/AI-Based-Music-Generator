# 🎶 Y.M.I.R – AI-Based Music Generator

**Y.M.I.R (Yielding Music for Internal Restoration)** is an AI-driven application that generates personalized music based on user emotions. By analyzing facial expressions, it crafts custom lyrics, composes instrumental tracks, and synthesizes vocals to produce a complete song tailored to the user's current mood.

---

## 📌 Features

- **Facial Emotion Recognition (FER):** Captures real-time facial expressions to determine the user's emotional state.
- **Lyrics Generation:** Creates contextually relevant lyrics aligning with detected emotions.
- **Instrumental Composition:** Produces music that complements the generated lyrics and mood.
- **AI Vocal Synthesis:** Utilizes advanced models like DiffSinger to render realistic singing vocals.
- **Integrated Web Interface:** Offers an intuitive platform for users to interact and experience personalized music generation.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS (located in the `templates/` directory)
- **Backend:** Python (Flask framework via `app.py`)
- **AI & ML Models:**
  - Emotion Detection: Deep learning models for FER
  - Lyrics & Music Generation: Custom NLP and music composition algorithms
  - Vocal Synthesis: Integration with DiffSinger for AI-generated vocals
- **Data Handling:** JSON-based knowledge base (`knowledge_base.json`)
- **Deployment:** Configured for platforms like Heroku using `Procfile`

---

## 🚀 Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/pallav110/AI-Based-Music-Generator.git
   cd AI-Based-Music-Generator
   ```

2. **Install Dependencies:**

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   ```bash
   python app.py
   ```

4. **Access the Web Interface:**

   Navigate to `http://localhost:5000` in your web browser to interact with Y.M.I.R.

---

## 🎤 Usage

1. **Emotion Detection:**

   - Upon accessing the web interface, allow camera permissions.
   - The system captures your facial expressions to determine your current emotion.

2. **Music Generation:**

   - Based on the detected emotion, Y.M.I.R generates appropriate lyrics and composes a matching instrumental track.

3. **Vocal Synthesis:**

   - The generated lyrics are converted into vocals using AI models, producing a complete song.

4. **Playback & Download:**

   - Listen to the personalized song directly on the platform.
   - Optionally, download the track for offline listening.

---

## 📁 Project Structure

```
AI-Based-Music-Generator/
├── app.py                  # Main Flask application
├── rag_module.py           # Retrieval-Augmented Generation module
├── test_rag.py             # Tests for RAG module
├── knowledge_base.json     # Data for RAG
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
├── templates/              # HTML templates
└── static/
    └── songs/              # Generated songs storage
```

---

## 🔮 Future Enhancements

- **Enhanced Emotion Detection:** Incorporate multimodal emotion recognition using voice and text inputs.
- **Expanded Music Styles:** Introduce diverse genres and styles for music generation.
- **User Profiles:** Allow users to save and revisit their generated songs.
- **Mobile Compatibility:** Optimize the platform for mobile device usage.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to enhance Y.M.I.R, please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
