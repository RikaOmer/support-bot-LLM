import os
import sys
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# --- הגדרות ---
DB_PATH = "chroma_db"
# מודלים משופרים - עדיין קלים למחשב:
# אפשרויות LLM (Ollama): "llama3.1" (מותקן), "mistral:7b", "phi3:mini", "llama3.2:3b"
OLLAMA_MODEL = "llama3.1"  # מותקן במחשב שלך, איכות מעולה
# אפשרויות Embedding: "intfloat/multilingual-e5-small" (קל יותר), "paraphrase-multilingual-mpnet-base-v2" (איכות טובה יותר)
HF_MODEL_NAME = "intfloat/multilingual-e5-small"  # קל מאוד (~130MB), מצוין לעברית

def format_docs(docs):
    """מעצב את המסמכים שנמצאו לפורמט ברור עם הפרדה"""
    if not docs:
        return "לא נמצא מידע רלוונטי במאגר הנתונים."
    
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        # הוספת מידע על המקור
        source = doc.metadata.get('source', 'לא ידוע')
        header_info = doc.metadata.get('Header 1', '')
        if header_info:
            source += f" > {header_info}"
        
        # הוספת התוכן עם הפרדה ברורה
        formatted_parts.append(
            f"[מקור {i}: {source}]\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(formatted_parts)

def start_chat():
    print("📂 טוען את הדאטה-בייס...")
    print(f"🔤 מודל Embedding: {HF_MODEL_NAME}")
    print(f"🤖 מודל LLM: {OLLAMA_MODEL}")
    embedding_function = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
    
    if not os.path.exists(DB_PATH):
        print("❌ שגיאה: הדאטה בייס לא נמצא.")
        return

    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    # הגדלנו את k ל-6 כדי לתפוס יותר הקשר
    # במקום similarity, נסה MMR לחיפוש מגוון יותר
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 12}  # בודק 12, מחזיר 6 הטובים ביותר
    )
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    template = """אתה עוזר טכני מקצועי.

הוראות חשובות:
1. השתמש רק במידע שמופיע ב-Context למטה. אל תמציא תשובות.
2. אם התשובה לא נמצאת ב-Context, כתוב בבירור: "המידע המבוקש לא נמצא במאגר הנתונים שלי."
3. כשמחפשים קוד, מספר טלפון, או מידע ספציפי - בדוק את הטבלאות והרשימות ב-Context בקפידה.
4. אם יש קישור לתמונה (כמו `![alt](path)`), כלול אותו בסוף התשובה.
5. תמיד ענה בעברית.
6. אם יש מספר מקורות רלוונטיים, ציין את המקור של כל חלק בתשובה.

Context:
{context}

שאלה:
{question}

תשובה:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # שרשרת RAG משופרת עם עיצוב נכון של המסמכים
    rag_chain = (
        {
            "context": retriever | format_docs,  # עיצוב המסמכים לפני שליחה ל-LLM
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🤖 הבוט מוכן! (כתוב 'exit' ליציאה)\n")
    
    while True:
        query = input("שאל אותי: ")
        if query.lower() in ['exit', 'quit', 'יציאה']:
            break
            
        print("\n🔍 מחפש (DEBUG MODE)...")
        docs = retriever.invoke(query)
        
        # --- הדפסת דיבאג: מה הבוט באמת רואה? ---
        if docs:
            print(f"✅ נמצאו {len(docs)} מקורות.")
            for i, doc in enumerate(docs[:3], 1):  # מציג רק 3 ראשונים
                source = doc.metadata.get('source', 'לא ידוע')
                header = doc.metadata.get('Header 1', '')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   [{i}] {source}" + (f" > {header}" if header else ""))
                print(f"       {preview}...")
            print()
        else:
            print("⚠️ לא נמצאו מסמכים.\n")

        print("💡 תשובה:")
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    start_chat()
