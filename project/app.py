from flask import Flask, request, jsonify, render_template
import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan dalam environment variables.")

class RecipeAssistant:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_prompt(self, ingredients):
        if not ingredients:
            return "Tolong beri daftar bahan makanan yang valid dalam format JSON yang diminta."

        return f"""
Saya ingin kamu menjadi asisten dapur.

Berikut daftar bahan yang saya miliki: {ingredients}

Tugasmu adalah:
- Periksa apakah bahan yang saya berikan termasuk bahan makanan yang bisa dimasak.
- Jika semua bahan **bukan bahan makanan**, jangan tampilkan apa pun.
- Jika ada minimal satu bahan makanan valid, berikan minimal 12 resep berbeda yang **menggunakan bahan-bahan yang saya miliki**.

⚠️ Jawaban HARUS hanya berupa JSON valid TANPA PENJELASAN TAMBAHAN, dalam format:
[
  {{
    "nama": "Nama masakan",
    "bahan": ["bahan 1", "bahan 2"],
    "langkah": ["Langkah 1", "Langkah 2"]
  }},
  ... (minimal 12 resep berbeda)
]
""".strip()

    def search_recipes(self, ingredients):
        prompt_text = self.generate_prompt(ingredients)
        print(f"📥 Bahan diterima: {ingredients}")

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_text)],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json"  # Ubah ke application/json
            )

            model = "learnlm-2.0-flash-experimental"
            result_text = ""

            # Generate content using the model
            for chunk in self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                result_text += chunk.text or ""

            print("📤 Output mentah dari Gemini:\n", result_text)

            # Extract JSON from the result
            json_match = re.search(r'\[\s*{.*?}\s*\]', result_text, re.DOTALL)
            if not json_match:
                raise ValueError("Tidak ditemukan format JSON yang valid dalam output Gemini.")

            json_str = json_match.group(0)
            print("🧾 Potongan JSON yang ditemukan:\n", json_str)
            parsed_json = json.loads(json_str)
            return parsed_json

        except Exception as e:
            print(f"❌ Terjadi error: {e}")
            raise

app = Flask(__name__)
assistant = RecipeAssistant(api_key)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_recipes():
    ingredients = request.json.get('ingredients')
    try:
        recipes = assistant.search_recipes(ingredients)
        return jsonify(recipes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_favorite', methods=['POST'])
def save_favorite():
    recipe_name = request.json.get('recipe_name')
    print(f"🔖 Menyimpan resep favorit: {recipe_name}")
    # Simpan resep favorit ke database atau file di sini
    return jsonify({'message': f"Resep '{recipe_name}' telah disimpan ke favorit!"})

if __name__ == "__main__":
    app.run(debug=True)
