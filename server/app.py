from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "server running"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def main():
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()