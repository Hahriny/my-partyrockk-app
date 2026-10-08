import json
import random
import boto3

BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")
IMAGE_MODEL_ID = "amazon.titan-image-generator-v2:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def handler(event, context):
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": "",
        }

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        body = {}

    topic = body.get("topic", "learning")
    image_type = body.get("image_type", "visual_aid")  # "visual_aid" or "topic_helper"

    if image_type == "topic_helper":
        prompt = (
            f"A cute cartoon illustration representing {topic} for a young learner. "
            "For Picture Matching show colorful animals and objects side by side. "
            "For Math Calculation show bright number blocks and counting fruits. "
            "For English and Sentences show a smiling child reading a big colorful book with floating letters. "
            "For Fun Quiz show a happy child with a glowing lightbulb. "
            "For Daily Living Skills show a cheerful child doing a helpful task. "
            "Warm bright colors, no text, soft outlines, children's picture book style."
        )
    else:
        prompt = (
            f"A bright and cheerful children's book style cartoon illustration for a child learning about {topic}. "
            "Very colorful with soft pastel tones, bold simple shapes, happy smiling characters, no text or letters anywhere. "
            "The scene feels warm, safe, and magical like a fairytale classroom. "
            "Include cute animals, stars, rainbows, or learning objects that match the topic."
        )

    seed = random.randint(0, 2147483647)

    # Titan Image Generator v2 request format
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "width": 1280,
            "height": 720,
            "seed": seed,
            "quality": "standard",
            "cfgScale": 8.0,
        },
    }

    try:
        response = BEDROCK.invoke_model(
            modelId=IMAGE_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        images = result.get("images", [])
        if not images:
            return {
                "statusCode": 200,
                "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
                "body": json.dumps({"image_b64": None, "placeholder": True, "message": "Image generation unavailable"}),
            }

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"image_b64": images[0]}),
        }
    except Exception as e:
        # Return a successful response with error info so the app doesn't break
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"image_b64": None, "placeholder": True, "message": f"Image generation unavailable: {str(e)[:150]}"}),
        }
