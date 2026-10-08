import json
import time
import uuid
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")
DYNAMODB = boto3.resource("dynamodb", region_name="ap-southeast-1")
TABLE = DYNAMODB.Table(os.environ.get("CHAT_TABLE_NAME", "learning-steps-chat-sessions"))

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
TTL_SECONDS = 24 * 60 * 60  # 24 hours

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def get_session_history(session_id):
    """Retrieve conversation history from DynamoDB."""
    try:
        response = TABLE.get_item(Key={"session_id": session_id})
        item = response.get("Item")
        if item:
            return item.get("history", [])
    except Exception:
        pass
    return []


def save_session_history(session_id, history):
    """Store conversation history to DynamoDB with 24-hour TTL."""
    ttl = int(time.time()) + TTL_SECONDS
    TABLE.put_item(
        Item={
            "session_id": session_id,
            "history": history,
            "ttl": ttl,
        }
    )


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    mood = data.get("mood", "Happy")
    topic = data.get("topic", "Picture Matching")
    lesson_text = data.get("lesson", "")
    message = data.get("message", "")

    # Get or create session ID
    session_id = data.get("session_id", "")
    if not session_id:
        session_id = str(uuid.uuid4())

    # Retrieve history from DynamoDB
    history = get_session_history(session_id)

    system_prompt = (
        f"You are a kind, friendly, and patient learning helper for an autistic child. "
        f"The child's name is {name} and they are {age} years old. "
        f"They are feeling {mood} today. "
        f"They are learning about {topic}. "
        f"Their lesson content is: {lesson_text}\n\n"
        "Always use simple, short sentences. Be warm, encouraging, and never make the child feel bad. "
        "Use emojis to make responses fun. Answer only one idea at a time. "
        "If the child seems confused, explain gently again in an even simpler way."
    )

    # Build messages from stored history + new user message
    messages = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": messages,
    }

    full_reply = []

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
                            text = delta.get("text", "")
                            full_reply.append(text)
                            yield text
        except Exception as e:
            yield f"\n\n❌ ERROR: {str(e)}"

    def stream_and_save():
        for chunk in generate():
            yield chunk

        # After streaming completes, save history to DynamoDB
        assistant_message = "".join(full_reply)
        if assistant_message:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assistant_message})
            # Keep last 20 turns to avoid oversized items
            trimmed = history[-20:]
            try:
                save_session_history(session_id, trimmed)
            except Exception:
                pass

    headers = dict(CORS_HEADERS)
    headers["X-Accel-Buffering"] = "no"
    headers["X-Session-Id"] = session_id
    return Response(stream_with_context(stream_and_save()),
                    content_type="text/plain; charset=utf-8",
                    headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
