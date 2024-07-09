from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Memuat model dan tokenizer dari direktori yang telah diekstrak
model_path = './trained_model'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Fungsi untuk membuat ringkasan
def summarize(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=length_penalty, 
        num_beams=num_beams, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def get_summary():
    text = request.form['text']
    summary = summarize(text)
    return render_template('index.html', summary=summary, text=text)

if __name__ == "__main__":
    app.run(debug=True)
