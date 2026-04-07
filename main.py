import io
from contextlib import redirect_stdout
from flask import Flask, Response

app = Flask(__name__)

@app.route("/")
def home():
    from inference import main
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main()
    output = buffer.getvalue()
    return Response(f"<pre>{output}</pre>", mimetype="text/html")

@app.route("/health")
def health():
    return {"status": "ok"}