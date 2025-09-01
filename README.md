# 🌐 Well Regards
## Project Overview
Well Regards is an experimental online chatroom that takes user-posted images and automatically generates descriptive sentences.
The system combines real-time chat functionality with image analysis to provide a unique, conversational experience.
## Features
- 🔹 Real-time Chatroom – Users can send text and images.
- 🔹 Image Analysis – Uploaded images are processed with OpenCV.
- 🔹 Automated Descriptions – Backend generates captions for images.
- 🔹 Future LLM Integration – Planned support for Gemini API to improve natural-language captions.
- 🔹 Modern Stack – Built with Python 3 backend and React frontend.
- 🔹 Supabase Integration – For authentication, database, and real-time syncing.
## 🛠️ Tech Stack

| Layer          | Technology Used                | Purpose |
|----------------|--------------------------------|---------|
| **Backend**    | Python 3 + uv                  | Dependency & environment management |
|                | OpenCV                         | Image analysis & preprocessing |
|                | Gemini API (planned)           | AI-based image-to-text generation |
| **Frontend**   | JavaScript (React)             | Web interface for chatroom |
| **Database**   | Supabase                       | User auth, chat history, storage |

## 📂 Project Structure
```
well-regards/
│── backend/
│   ├── main.py
│   ├── image_handler.py # OpenCV image analysis
│   ├── requirements.txt
│
│── frontend/
│   ├── src/
│   │   ├── App.jsx      # React app
│   │   ├── components/  
│   │   ├── pages/       
│
│── supabase/            # SQL migrations, config
│── README.md
```
## 🚀 Setup Instructions
1. Backend
```
uv venv
uv add -r requirements.txt
```
2. Frontend
```
cd frontend
npm install
npm run dev
```
3. Supabase
- Create a project on Supabase
- Enable authentication + storage buckets (for image uploads).
- Connect with backend via API keys.

# 🧪 Usage Flow

- User joins chatroom.
- User uploads image.
- Backend (Python + OpenCV) processes image.
- Descriptive text is generated. (LLM fallback planned via Gemini API).
- Both image + description are displayed in chatroom.

## Team
This project is made by the *Banana Byte Entertainment* team.

<a href="https://github.com/Banana-Byte-Entertainment/well-regards/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Banana-Byte-Entertainment/well-regards" alt="contrib.rocks image" />
</a>

## License
This project is provided for educational and portfolio purposes. Please [contact the authors](mailto:bananabyteentertainment@gmail.com) for inquiries about reuse or distribution.