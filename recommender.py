from transformers import pipeline

class EmotionRecommender:
    def __init__(self):
        """Initializes the model and label mappings."""
        print("Loading AI Model (This may take a moment)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        )
        
        self.CORE_LABELS = [
            "joyful", "trusting", "fearful", "surprised",
            "sad", "disgusted", "angry", "anticipating"
        ]
        
        self.LABEL_MAP = {
            "joyful": "joy",
            "trusting": "trust",
            "fearful": "fear",
            "surprised": "surprise",
            "sad": "sadness",
            "disgusted": "disgust",
            "angry": "anger",
            "anticipating": "anticipation"
        }
        
        self.HYPOTHESIS_TEMPLATE = "The person saying this is feeling {}."

    def apply_rules(self, text, scores_dict):
        """Applies keyword-based boosts, clamping the maximum probability to 1.0."""
        text_lower = text.lower()

        # Anticipation (future intent)
        if any(word in text_lower for word in ["will", "going to", "about to", "soon", "later", "plan", "expect"]):
            scores_dict["anticipating"] = min(scores_dict.get("anticipating", 0) + 0.15, 1.0)

        # Trust (confidence / certainty)
        if any(word in text_lower for word in ["definitely", "for sure", "guarantee", "reliable", "trust"]):
            scores_dict["trusting"] = min(scores_dict.get("trusting", 0) + 0.15, 1.0)

        # Fear (anxiety language)
        if any(word in text_lower for word in ["nervous", "worried", "anxious", "scared", "panic"]):
            scores_dict["fearful"] = min(scores_dict.get("fearful", 0) + 0.20, 1.0)

        # Joy (strong positive cues)
        if any(word in text_lower for word in ["happy", "excited", "amazing", "great", "awesome", "crushed it"]):
            scores_dict["joyful"] = min(scores_dict.get("joyful", 0) + 0.15, 1.0)

        return scores_dict

    def analyze_mood(self, text, threshold=0.35):
        """Analyzes text and returns the top Plutchik emotions."""
        result = self.classifier(
            text,
            self.CORE_LABELS,
            multi_label=True,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE
        )

        # Convert to dictionary
        scores_dict = {
            result['labels'][i]: result['scores'][i]
            for i in range(len(result['labels']))
        }

        # Apply hybrid rules
        scores_dict = self.apply_rules(text, scores_dict)

        # Filter by threshold
        valid_emotions = [
            (label, score)
            for label, score in scores_dict.items()
            if score > threshold
        ]

        # Fallback: grab the highest score from the MODIFIED dict, not the raw result
        if not valid_emotions:
            top_label = max(scores_dict, key=scores_dict.get)
            top_score = scores_dict[top_label]
            valid_emotions.append((top_label, top_score))

        # Sort descending
        valid_emotions.sort(key=lambda x: x[1], reverse=True)

        # Top 3 only
        top_emotions = valid_emotions[:3]

        # Normalize labels to Plutchik format
        detected_emotions = {
            self.LABEL_MAP[label]: round(score, 3)
            for label, score in top_emotions
        }

        return detected_emotions

# ==========================================
# TEST WRAPPER
# ==========================================
if __name__ == "__main__":
    # Initialize the engine once
    recommender = EmotionRecommender()
    
    while True:
        enter_text = input("\nEnter text to analyze (or type 'n' to exit): ")

        if enter_text.lower() != 'n':
            print("\n--- NEW INPUT ---")
            print(f"Text: '{enter_text}'")
            emotions = recommender.analyze_mood(enter_text)
            print(f"Detected Emotions: {emotions}")
        else:
            print("Exiting...")
            break

