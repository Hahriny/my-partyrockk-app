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
def parent_notes():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    topic = data.get("topic", "Picture Matching")
    mood = data.get("mood", "Happy")

    prompt = f"""You are a specialist in autism education and child development. Write a helpful and warm support guide for the parent or teacher of this learner.

Learner name: {name}
Learner age: {age}
Today's topic: {topic}
Learner mood today: {mood}

Create a structured guide with these sections:

📋 TODAY'S LESSON SUMMARY
- 2 short sentences summarizing what was practiced today

💡 HOW TO SUPPORT AT HOME OR IN CLASS
- 3 simple practical tips tailored to the topic and age
- Each tip is 1–2 sentences

🧠 UNDERSTANDING THE MOOD
- 2 gentle suggestions for responding to the learner's mood today

🌱 NEXT STEP SUGGESTIONS
- 2 follow-up activities or topics to try next time

💛 ENCOURAGEMENT FOR YOU
- A warm 2-sentence thank-you to the caregiver

Keep the tone warm, professional, and jargon-free."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
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
