import os
import shutil
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- הגדרות ---
# עדכן את הנתיב לפי המבנה שלך
DATA_PATH = os.path.join(os.path.dirname(__file__), "markdown")
DB_PATH = "chroma_db"
# מודל Embedding משופר - קל יותר ואיכותי יותר לעברית
# אפשרויות: "intfloat/multilingual-e5-small" (קל, ~130MB), "paraphrase-multilingual-mpnet-base-v2" (איכות טובה יותר, ~420MB)
HF_MODEL_NAME = "intfloat/multilingual-e5-small"  # קל מאוד, מצוין לעברית

def build_database():
    # 1. ניקוי DB ישן
    if os.path.exists(DB_PATH):
        print("🗑️ מוחק דאטה-בייס ישן...")
        shutil.rmtree(DB_PATH)

    # 2. קריאת קבצים
    if not os.path.exists(DATA_PATH):
        print("❌ שגיאה: תיקיית הדאטה לא קיימת.")
        print(f"   נתיב נדרש: {DATA_PATH}")
        return

    print("📖 קורא קבצי Markdown ומבצע חלוקה חכמה...")
    
    # הגדרת החלוקה לפי כותרות - זה הקסם!
    # זה שומר את הכותרת כחלק מהמידע של כל מקטע
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    all_header_splits = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.md')]
    
    for filename in files:
        file_path = os.path.join(DATA_PATH, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # חלוקה ראשונית לפי כותרות
                md_header_splits = markdown_splitter.split_text(content)
                
                # הוספת שם הקובץ למטא-דאטה
                for doc in md_header_splits:
                    doc.metadata["source"] = filename
                    # שילוב הכותרות לתוך הטקסט עצמו כדי שהמודל יראה אותן בבירור
                    header_context = " > ".join([v for k, v in doc.metadata.items() if k.startswith("Header")])
                    if header_context:
                        doc.page_content = f"נושא: {header_context}\n\n{doc.page_content}"
                
                all_header_splits.extend(md_header_splits)
                
        except Exception as e:
            print(f"   ❌ נכשל בטעינת {filename}: {e}")

    print(f"✅ נוצרו {len(all_header_splits)} מקטעים מבוססי כותרות.")

    # 3. חלוקה משנית (אם יש מקטעים ארוכים מדי גם אחרי החלוקה לכותרות)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    final_chunks = text_splitter.split_documents(all_header_splits)
    print(f"✂️ חלוקה סופית ל-{len(final_chunks)} מקטעים.")

    # 4. יצירת ה-Vector DB
    print("🚀 בונה אינדקס...")
    print(f"🔤 משתמש במודל Embedding: {HF_MODEL_NAME}")
    embedding_function = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
    
    Chroma.from_documents(
        documents=final_chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print("✨ הדאטה בייס החדש מוכן!")

if __name__ == "__main__":
    build_database()
