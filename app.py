from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the model inside the route to manage memory better
def get_summarizer():
    return pipeline(
        "summarization",
        model="t5-small",  # Smaller model for lower memory usage
        tokenizer="t5-small",
        device=-1,
        framework="pt"
    )

def split_text(text, max_length=512):
    """Splits the text into smaller chunks if it exceeds the max_length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = ""
    if request.method == 'POST':
        article = request.form['article']
        max_length = int(request.form['max_length'])
        min_length = max_length // 2
        
        summarizer = get_summarizer()  # Initialize the summarizer when needed
        article_chunks = split_text(article, max_length)  # Split the article into smaller chunks

        # Process each chunk and concatenate summaries
        summaries = []
        for chunk in article_chunks:
            result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=True)
            summaries.append(result[0]['summary_text'])
        
        summary = ' '.join(summaries)  # Combine summaries of all chunks

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
