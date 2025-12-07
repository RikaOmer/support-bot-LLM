import os
import shutil
from markdownify import markdownify as md

# --- הגדרות נתיבים ---
# עדכן את הנתיבים לפי המבנה שלך
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "wordFilter")
OUTPUT_DIR = os.path.join(BASE_DIR, "markdown")

def convert_htmls():
    # יצירת תיקיית יעד אם לא קיימת
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # מעבר על כל הקבצים בתיקיית הקלט
    try:
        files = os.listdir(INPUT_DIR)
    except FileNotFoundError:
        print(f"שגיאה: התיקייה {INPUT_DIR} לא נמצאה.")
        return

    for filename in files:
        if filename.lower().endswith((".htm", ".html")):
            print(f"מעבד את הקובץ: {filename}...")
            
            base_name = os.path.splitext(filename)[0]
            file_path = os.path.join(INPUT_DIR, filename)
            
            html_content = None
            # רשימת קידודים לניסיון - UTF-16 הוא ברירת המחדל של Word לקבצי HTM
            encodings = ['utf-16', 'utf-8', 'cp1255', 'windows-1255', 'iso-8859-8']

            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        html_content = f.read()
                    # אם הצלחנו לקרוא, יוצאים מהלולאה
                    break 
                except UnicodeDecodeError:
                    continue
            
            if html_content is None:
                print(f"   ERROR: לא ניתן לפענח את הקידוד של הקובץ {filename}. מדלג.")
                continue

            # 2. המרה למרקדאון
            try:
                markdown_text = md(html_content, heading_style="ATX")
                
                # ניקוי שורות ריקות מיותרות
                clean_md = "\n".join([line for line in markdown_text.splitlines() if line.strip()])
                
                # הוספת כותרת מקור
                final_content = f"# מקור: {base_name}\n\n{clean_md}"

                # 3. שמירת קובץ ה-MD
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_content)

                # 4. העתקת תיקיית התמונות (אם קיימת)
                expected_folder_name = f"{base_name}_files"
                src_folder = os.path.join(INPUT_DIR, expected_folder_name)
                dst_folder = os.path.join(OUTPUT_DIR, expected_folder_name)
                
                if os.path.exists(src_folder):
                    if os.path.exists(dst_folder):
                        shutil.rmtree(dst_folder)
                    shutil.copytree(src_folder, dst_folder)
                    print(f"   -> הועתקה תיקיית תמונות: {expected_folder_name}")
            
            except Exception as e:
                print(f"   ERROR בעת המרה/שמירה של {filename}: {e}")

    print("\n--- הסתיים בהצלחה! ---")
    print(f"הקבצים המוכנים לבוט נמצאים בתיקייה: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_htmls()
