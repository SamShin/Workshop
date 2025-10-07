import boto3
import json
from pathlib import Path

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')


def detect_text(image_path):
    """
    Detect and extract text from images
    """
    print(f"\nExtracting text from: {image_path}")
    print("-" * 50)

    with open(image_path, 'rb') as image:
        response = rekognition.detect_text(
            Image={'Bytes': image.read()}
        )

    text_detections = response['TextDetections']

    if not text_detections:
        print("No text detected")
        return response

    # Separate by type (LINE vs WORD)
    lines = [t for t in text_detections if t['Type'] == 'LINE']
    words = [t for t in text_detections if t['Type'] == 'WORD']

    print(f"Found {len(lines)} line(s) and {len(words)} word(s)\n")

    print("Detected Text (by line):")
    for line in lines:
        confidence = line['Confidence']
        text = line['DetectedText']
        print(f"  • \"{text}\" ({confidence:.1f}% confident)")

    return response

# Example usage
detect_text('image/text.png')