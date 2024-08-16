import streamlit as st
import os
from beyondllm import source, embeddings, retrieve, llms, generator
import config

# API Anahtarlarını config dosyasından yükleme
os.environ['HF_TOKEN'] = config.HF_TOKEN
os.environ['GOOGLE_API_KEY'] = config.GOOGLE_API_KEY
os.environ['HUGGINGFACE_ACCESS_TOKEN'] = os.environ['HF_TOKEN']

# Uygulama başlığı
st.title("RAG Pipeline with BeyondLLM")

# Kullanıcıdan YouTube video linkini alma
video_url = st.text_input("Enter the YouTube video URL:", "https://www.youtube.com/watch?v=ZM1bdh2mDJQ")

# Veri yükleme ve embedding işlemleri
if st.button("Process Video"):
    with st.spinner("Processing..."):
        data = source.fit(
            path=video_url,
            dtype="youtube",
            chunk_size=1024,
            chunk_overlap=0
        )
        
        model_name = 'BAAI/bge-small-en-v1.5'
        embed_model = embeddings.HuggingFaceEmbeddings(model_name=model_name)
        
        retriever = retrieve.auto_retriever(
            data=data,
            embed_model=embed_model,
            type="cross-rerank",
            mode="OR",
            top_k=2
        )
        
        st.success("Video processed successfully!")

# Kullanıcıdan sorgu alma
question = st.text_input("Enter your question:")

# Model ve sorgu sonuçlarını gösterme
if st.button("Get Answer"):
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
    
    # retriever doğru sırada kullanılmalı
    if 'retriever' in locals():
        pipeline = generator.Generate(
            question=question,
            retriever=retriever,
            system_prompt=system_prompt,
            llm=llm
        )
        
        response = pipeline.call()
        st.write("Model yanıtı:", response)
        
        # RAG Triad değerlendirme sonuçlarını gösterme
        rag_evals = pipeline.get_rag_triad_evals()
        st.write("RAG Triad Değerlendirmesi:", rag_evals)
    else:
        st.error("Please process the video before asking a question.")
