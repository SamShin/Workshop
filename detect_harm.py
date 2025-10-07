import boto3
import json
from pathlib import Path

rekognition = boto3.client('rekognition', region_name='us-west-2')


def detect_moderation_labels(image_path):
    """
    Detect potentially inappropriate content
    """
    print(f"\nChecking content in: {image_path}")
    print("-" * 50)

    with open(image_path, 'rb') as image:
        response = rekognition.detect_moderation_labels(
            Image={'Bytes': image.read()},
            MinConfidence=60
        )

    moderation_labels = response['ModerationLabels']

    if not moderation_labels:
        print("No inappropriate content detected")
    else:
        print(f"Found {len(moderation_labels)} potential issue(s):")
        for label in moderation_labels:
            print(f"  • {label['Name']}: {label['Confidence']:.1f}% confident")
            if label.get('ParentName'):
                print(f"    Category: {label['ParentName']}")

    return response

# Example usage
detect_moderation_labels('image/game.jpg')