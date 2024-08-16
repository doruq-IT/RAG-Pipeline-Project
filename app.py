import streamlit as st
import os
from beyondllm import source, embeddings, retrieve, llms, generator
import config

# API Anahtarlarını config dosyasından yükleme
os.environ['HF_TOKEN'] = config.HF_TOKEN
os.environ['GOOGLE_API_KEY'] = config.GOOGLE_API_KEY
os.environ['HUGGINGFACE_ACCESS_TOKEN'] = os.environ['HF_TOKEN']

# Uygulama başlığı ve giriş bölümü
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #0073e6;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
        }
        .input-section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .text-input {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #dddddd;
            margin-bottom: 20px;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
        }
        .button {
            display: inline-block;
            width: 100%;
            background-color: #0073e6;
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #005bb5;
        }
        .result-section {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .highlight {
            background-color: #ffefc3;
            padding: 5px;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>RAG Pipeline with BeyondLLM</h1>", unsafe_allow_html=True)

# Kullanıcıdan YouTube video linkini alma
video_url = st.text_input("Enter the YouTube video URL:", "https://www.youtube.com/watch?v=ZM1bdh2mDJQ", key="video_url", label_visibility="collapsed")

# Veri yükleme ve embedding işlemleri
if st.button("Process Video", key="process_button"):
    with st.spinner("Processing..."):
        data = source.fit(
            path=video_url,
            dtype="youtube",
            chunk_size=1024,
            chunk_overlap=0
        )
        
        model_name = 'BAAI/bge-small-en-v1.5'
        embed_model = embeddings.HuggingFaceEmbeddings(model_name=model_name)
        
        # Retriever'ı session state'e kaydetme
        st.session_state['retriever'] = retrieve.auto_retriever(
            data=data,
            embed_model=embed_model,
            type="cross-rerank",
            mode="OR",
            top_k=2
        )
        
        st.success("Video processed successfully!")

# Kullanıcıdan sorgu alma
question = st.text_input("Enter your question:", key="question", label_visibility="collapsed")

# Model ve sorgu sonuçlarını gösterme
if st.button("Get Answer", key="answer_button"):
    if 'retriever' in st.session_state:
        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
        
        llm = llms.HuggingFaceHubModel(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.environ.get('HF_TOKEN')
        )
        
        system_prompt = f"""
        <s>[INST]
        You are an AI Assistant.
        Please provide direct answers to questions.
        [/INST]
        </s>
        """
        
        pipeline = generator.Generate(
            question=question,
            retriever=st.session_state['retriever'],
            system_prompt=system_prompt,
            llm=llm
        )
        
        response = pipeline.call()
        st.markdown(f"<p><strong>Model yanıtı:</strong> <span class='highlight'>{response}</span></p>", unsafe_allow_html=True)
        
        # RAG Triad değerlendirme sonuçlarını gösterme
        rag_evals = pipeline.get_rag_triad_evals()
        st.markdown(f"<p><strong>RAG Triad Değerlendirmesi:</strong> <span class='highlight'>{rag_evals}</span></p>", unsafe_allow_html=True)
    else:
        st.error("Please process the video before asking a question.")
