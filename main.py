import io
from contextlib import redirect_stdout
from flask import Flask, jsonify, request, Response
from pydantic import ValidationError

from app.env import GDMultiEvalEnv
from app.models import GDAction

app = Flask(__name__)

env = GDMultiEvalEnv(difficulty="hard")
initialized = False


@app.get("/")
def home():
    from inference import main as inference_main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        inference_main()

    output = buffer.getvalue()

    return Response(f"<pre>{output}</pre>", mimetype="text/html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/reset")
def reset_env():
    global initialized
    obs = env.reset()
    initialized = True
    return jsonify(obs.model_dump()), 200


@app.get("/state")
def get_state():
    global initialized
    if not initialized:
        obs = env.reset()
        initialized = True
        return jsonify({
            "message": "Environment was not initialized. Auto-reset performed.",
            "observation": obs.model_dump(),
            "state": env.state().model_dump()
        }), 200

    return jsonify(env.state().model_dump()), 200


@app.post("/step")
def step_env():
    global initialized

    if not initialized:
        env.reset()
        initialized = True

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        action = GDAction(**data)
    except ValidationError as e:
        return jsonify({
            "error": "Invalid action payload",
            "details": e.errors()
        }), 400

    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }), 200


def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()