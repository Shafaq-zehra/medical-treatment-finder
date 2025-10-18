# medical-treatment-finder
AI-powered Streamlit app that fetches PubMed medical research papers, extracts treatment-focused information, and generates answers and  treatment snippets using transformer-based models and FAISS retrieval.rieval.
Features:
1. Fetches real PubMed studies automatically
2. Focuses on treatment-related medical research
3. Uses FAISS for semantic search of abstracts
4. Generates detailed summaries using google/flan-t5-large
5. Beautiful and interactive Streamlit interface
6. 100% Python — easy to run or deploy

Tech Stack:
Component	Technology
Frontend	Streamlit
Backend / AI	Hugging Face Transformers (Flan-T5)
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Search	FAISS vector similarity
Data Source	PubMed API (NCBI eUtils)

Installation:
Clone the repository and install dependencies:

git clone https://github.com/<your-username>/medical-treatment-finder.git
cd medical-treatment-finder
pip install -r requirements.txt

 Run the App Locally
streamlit run app.py

Then open your browser at http://localhost:8501

🌐 Deploy on Streamlit Cloud
Push your project to GitHub
Visit https://share.streamlit.io
Click “New App” → Choose your repo → Select app.py
Click Deploy
Your app will go live at:
https://<your-username>-medical-treatment-finder.streamlit.app

🧠 Example Usage



Query:

"Heart disease"




Question:

"What are the latest treatments and therapies available for heart disease?"




Output:

The model summarizes the latest PubMed abstracts, highlighting effective drugs, therapies, and ongoing clinical interventions.

📁 Folder Structure
medical-treatment-finder/
│
├── app.py                # Streamlit interface
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignore cache, models, etc.

🧑‍⚕️ How It Works
PubMed Fetching: Uses NCBI E-utilities API to fetch the latest papers
Chunking: Breaks abstracts into smaller, treatment-focused text segments
Embedding & Indexing: Creates FAISS embeddings for semantic similarity search
Answer Generation: Generates comprehensive treatment summaries via flan-t5-large
🧩 Example Topics



Try queries like:

“Diabetes treatment”
“Cancer therapy”
“COVID-19 drug intervention”
“Heart failure management”
🧑‍💻 Author



Shafaq Zehra
AI & Data Science in Healthcare | Intelligent Systems Developer

🏥 Disclaimer



This application is for educational and research purposes only.
It does not replace professional medical advice. Always consult a healthcare professional before making treatment decisions.
