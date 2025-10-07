import boto3
import json
from pathlib import Path

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')

def detect_labels(image_path):
    """
    Detect objects, scenes, and concepts in an image
    """
    print(f"\n Analyzing: {image_path}")
    print("-" * 50)

    # Read image file
    with open(image_path, 'rb') as image:
        response = rekognition.detect_labels(
            Image={'Bytes': image.read()},
            MaxLabels=10,
            MinConfidence=75
        )

    # Display results
    print("Detected Labels:")
    for label in response['Labels']:
        confidence = label['Confidence']
        name = label['Name']
        print(f"  • {name}: {confidence:.1f}% confident")

        # Show specific instances if available
        if label.get('Instances'):
            print(f"    Found {len(label['Instances'])} instance(s)")

    return response

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    detect_labels('image/room.jpg')