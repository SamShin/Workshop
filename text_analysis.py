import boto3
import workshop_config as config

# Initialize Comprehend client
comprehend = boto3.client(
    'comprehend',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_REGION
)

def analyze_sentiment(text):
    print(f"\nAnalyzing sentiment for text ({len(text)} characters)...")
    print("-" * 60)

    response = comprehend.detect_sentiment(
        Text=text,
        LanguageCode='en'
    )

    sentiment = response['Sentiment']
    scores = response['SentimentScore']

    print(f"Overall Sentiment: {sentiment}")
    print(f"\nConfidence Scores:")
    print(f"  Positive: {scores['Positive']*100:.1f}%")
    print(f"  Negative: {scores['Negative']*100:.1f}%")
    print(f"  Neutral:  {scores['Neutral']*100:.1f}%")
    print(f"  Mixed:    {scores['Mixed']*100:.1f}%")

    return response

def detect_entities(text):
    print(f"\nDetecting named entities...")
    print("-" * 60)

    response = comprehend.detect_entities(
        Text=text,
        LanguageCode='en'
    )

    # Group entities by type for cleaner display
    entities_by_type = {}
    for entity in response['Entities']:
        entity_type = entity['Type']
        confidence = entity['Score']
        text_value = entity['Text']

        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append({
            'text': text_value,
            'confidence': confidence
        })

    # Display results organized by entity type
    for entity_type, items in sorted(entities_by_type.items()):
        print(f"\n{entity_type}:")
        for item in items:
            print(f"  - {item['text']} ({item['confidence']*100:.0f}% confident)")

    return response

def detect_key_phrases(text):
    print(f"\nExtracting key phrases...")
    print("-" * 60)

    response = comprehend.detect_key_phrases(
        Text=text,
        LanguageCode='en'
    )

    print(f"Found {len(response['KeyPhrases'])} key phrases:\n")

    # Sort by confidence score
    sorted_phrases = sorted(
        response['KeyPhrases'],
        key=lambda x: x['Score'],
        reverse=True
    )

    for i, phrase in enumerate(sorted_phrases[:10], 1):
        score = phrase['Score']
        text = phrase['Text']
        print(f"  {i}. {text} (confidence: {score*100:.0f}%)")

    return response

def detect_language(text):
    response = comprehend.detect_dominant_language(Text=text)

    print(f"\nLanguage Detection:")
    print("-" * 60)
    for lang in response['Languages'][:3]:
        code = lang['LanguageCode']
        score = lang['Score']
        print(f"  {code}: {score*100:.1f}% confident")

    return response

def analyze_complete(text):
    print("\n" + "="*60)
    print("AWS COMPREHEND - COMPLETE TEXT ANALYSIS")
    print("="*60)
    print(f"\nInput Text:\n{text[:200]}{'...' if len(text) > 200 else ''}\n")

    # Run all analyses
    detect_language(text)
    analyze_sentiment(text)
    detect_entities(text)
    detect_key_phrases(text)

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)

# Example usage
if __name__ == "__main__":

    # Example 1: Customer Review Analysis
    print("\n### EXAMPLE 1: CUSTOMER REVIEW ANALYSIS ###")
    review = """
    I recently stayed at the Hilton Seattle for a business conference.
    The location was perfect, right downtown near Pike Place Market.
    However, the WiFi was extremely slow, which made it difficult to
    work from my room. The breakfast buffet was excellent though, and
    the staff were very professional. Overall, I'd give it 3 out of 5 stars.
    """
    analyze_complete(review)

    # Example 2: News Article Excerpt
    print("\n\n### EXAMPLE 2: NEWS ARTICLE ANALYSIS ###")
    news = """
    The University of Washington announced a $50 million investment in
    artificial intelligence research on Tuesday. President Robert J. Jones
    stated that the funding will support the Paul G. Allen School of Computer
    Science & Engineering's expansion. The initiative aims to hire 20 new
    faculty members over the next three years.
    """
    analyze_complete(news)

    # Example 3: Social Media Content
    print("\n\n### EXAMPLE 3: SOCIAL MEDIA POST ###")
    social = """
    Just finished the most amazing hike at Mount Rainier National Park!
    The weather was perfect and the views were absolutely breathtaking.
    Definitely recommend the Skyline Trail if you're in the Seattle area.
    """
    analyze_complete(social)

    # Example 4: Multi-Lingual Text
    print("\n\n### EXAMPLE 3: Multi-Lingual Text ###")
    language = """
    It's a beautiful day to learn something new.
    Es un día hermoso para aprender algo nuevo.
    C'est une belle journée pour apprendre quelque chose de nouveau.
    """

    analyze_complete(language)

    print("\n\n" + "="*60)
    print("Demo complete. Try analyzing your own text!")
    print("="*60)