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
def writing_practice():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    topic = data.get("topic", "Picture Matching")

    prompt = f"""You are a kind and encouraging writing coach for autistic learners.

The learner's name is {name} and they are {age} years old.
The current lesson topic is: {topic}

Create a fun FILL IN THE BLANKS writing practice. Use ___ (three underscores) for each blank the learner needs to fill in.

▶ For AGE 5–10:
- Create 4 simple sentences related to the topic
- Each sentence must have ONE blank (___) where a word is missing
- After each sentence, show the correct answer in parentheses
- Example format: "The cat is ___ (soft)"
- Use simple words and emojis

▶ For AGE 11–19:
- Create 4 sentences related to {topic} with ONE blank (___) each
- Make sentences slightly longer and more challenging
- After each sentence, show the correct answer in parentheses
- Include a short grammar tip at the end

IMPORTANT: Always use exactly ___ (three underscores) for blanks. Keep the tone warm and celebratory."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.3,
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
