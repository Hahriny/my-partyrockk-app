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
def generate_lesson():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    topic = data.get("topic", "Picture Matching")
    mood = data.get("mood", "Happy")

    prompt = f"""You are a warm, patient, and encouraging teacher who specializes in supporting autistic learners of all ages. Your lessons must always feel safe, fun, and achievable.

The learner's name is {name} and they are {age} years old.
The selected topic is: {topic}
The learner is feeling: {mood}

First, write one short kind sentence acknowledging how they feel today and encouraging them gently.

---

🌟 IMPORTANT: Adapt your lesson style based on age:

▶ For AGE 5–10 (Young Learners):
- Use VERY short sentences (3–6 words each)
- Use BIG emojis frequently
- Use simple everyday words only
- Focus on colors, shapes, sizes, and pictures
- Use repetition and praise after every step
- Keep instructions to one action at a time

▶ For AGE 11–19 (Older Learners):
- Use clear but slightly longer sentences
- Use emojis but less frequently
- Introduce vocabulary building and thinking exercises
- Encourage independent thinking
- Use structured sections with headers
- Be encouraging but treat them with respect

---

Create the lesson based on the topic:

🎨 If Picture Matching:
- Ages 5–10: Name 4 colorful animals or objects. Describe each with color, shape, and size in 2 simple sentences. Use lots of emojis. After each write: Can you find this picture? 🔍
- Ages 11–19: Name 4 objects or animals. Write 3 descriptive sentences each. Ask the learner to write one describing sentence.

🔢 If Math Calculation:
- Ages 5–10: Create 4 addition or subtraction problems with numbers 1–10. Use emoji visuals to show numbers. Walk through each answer step by step.
- Ages 11–19: Create 4 problems mixing addition, subtraction, and multiplication up to 50. Show working steps. Include one real-life word problem.

📝 If English and Sentences:
- Ages 5–10: Show 4 simple words with meaning and emoji. Use each in a 4-word sentence. Ask the learner to repeat each.
- Ages 11–19: Introduce 4 vocabulary words with definitions and example sentences. Give a fill-in-the-blank exercise and a creative writing prompt.

❓ If Fun Quiz:
- Ages 5–10: Create 4 simple questions about colors, animals, or shapes. Give 2 answer choices each. Celebrate every answer.
- Ages 11–19: Create 4 multiple choice questions with 3 choices labeled A, B, C. Reveal answers with short explanations.

🌟 If Daily Living Skills:
- Ages 5–10: Pick one skill like washing hands. Break into 5 simple steps with one action word and emoji each.
- Ages 11–19: Pick an advanced skill like preparing a snack. Break into 6 steps with brief explanations and a tip.

---

Always end with:
✅ You did a great job today! Keep going — you are absolutely amazing! 🌟💛"""

    content = [{"type": "text", "text": prompt}]

    # Handle optional file upload
    file_data = data.get("file_data")
    file_mime = data.get("file_mime", "")
    if file_data:
        if file_mime.startswith("image/"):
            media_type = file_mime
            content.insert(0, {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": file_data}
            })
        else:
            content.insert(0, {
                "type": "document",
                "source": {"type": "base64", "media_type": file_mime, "data": file_data}
            })

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
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
