from transformers import T5Tokenizer, T5ForConditionalGeneration

# Memuat model dan tokenizer dari direktori yang telah diekstrak
model_path = './trained_model'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Fungsi untuk membuat ringkasan
def summarize(text, max_length=1024, min_length=200, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=length_penalty, 
        num_beams=num_beams, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



# Contoh penggunaan
if __name__ == "__main__":
    article_text = """Masukkan teks artikel di sini. Ini adalah contoh artikel panjang yang akan diringkas oleh model. Pastikan artikel ini cukup panjang untuk menghasilkan ringkasan yang bermakna."""
    summary = summarize(article_text)
    print("Ringkasan:", summary)
