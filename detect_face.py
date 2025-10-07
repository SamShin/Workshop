import boto3
import json
from pathlib import Path

# Initialize Rekognition client
rekognition = boto3.client('rekognition', region_name='us-west-2')

def detect_faces(image_path):
    """
    Detect faces and analyze attributes (age, emotion, etc.)
    """
    print(f"\nAnalyzing faces in: {image_path}")
    print("-" * 50)

    with open(image_path, 'rb') as image:
        response = rekognition.detect_faces(
            Image={'Bytes': image.read()},
            Attributes=['ALL']  # Get all face attributes
        )

    face_count = len(response['FaceDetails'])
    print(f"Found {face_count} face(s)\n")

    for i, face in enumerate(response['FaceDetails'], 1):
        print(f"Face #{i}:")

        # Age range
        age_low = face['AgeRange']['Low']
        age_high = face['AgeRange']['High']
        print(f"  • Age: {age_low}-{age_high} years")

        # Gender
        gender = face['Gender']['Value']
        gender_conf = face['Gender']['Confidence']
        print(f"  • Gender: {gender} ({gender_conf:.1f}% confident)")

        # Emotions
        emotions = sorted(face['Emotions'],
                         key=lambda x: x['Confidence'],
                         reverse=True)[:3]
        print(f"  • Top emotions:")
        for emotion in emotions:
            print(f"    - {emotion['Type']}: {emotion['Confidence']:.1f}%")

        # Other attributes
        print(f"  • Smile: {face['Smile']['Value']}")
        print(f"  • Eyes open: {face['EyesOpen']['Value']}")
        print(f"  • Wearing glasses: {face['Eyeglasses']['Value']}")
        print()

    return response

# Example usage
detect_faces('image/group_photo.jpg')