from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = ""
    if request.method == 'POST':
        article = request.form['article']
        max_length = int(request.form['max_length'])
        min_length = max_length // 2  # Example: min_length is half of max_length
        summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=True)
        summary = summary[0]['summary_text']
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
