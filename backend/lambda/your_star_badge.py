import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def star_badge():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    topic = data.get("topic", "Picture Matching")
    mood = data.get("mood", "Happy")

    prompt = f"""You are the most cheerful reward system for autistic learners.

The learner's name is {name} and they are {age} years old.
They completed a lesson on: {topic}
They were feeling: {mood}

Create a fun reward celebration with:
1. 🏆 A personalized congratulations message using their name
2. 🌟 A special badge name based on the topic (e.g. Math Star, Reading Hero, Picture Champion)
3. 🎨 3 to 4 celebration lines with big emojis and short energetic sentences
4. 💛 A kind note about how brave they were based on their mood today
5. ⭐ End with: You are a SUPERSTAR! See you next time!

For ages 5–10: Maximum emojis, super short words, very colorful and exciting!
For ages 11–19: Fun but slightly more mature, like a real achievement certificate."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    def generate():
        try:
            response = BEDROCK.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            for event in response["body"]:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"].decode())
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
        except Exception as e:
            yield f"\n\n❌ ERROR: {str(e)}"

    headers = dict(CORS_HEADERS)
    headers["X-Accel-Buffering"] = "no"
    return Response(stream_with_context(generate()),
                    content_type="text/plain; charset=utf-8",
                    headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
