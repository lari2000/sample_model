import torch
from transformers import pipeline
import warnings

# Mute the specific fallback warning for DirectML
warnings.filterwarnings("ignore", message=".*aten::unique_consecutive.*")

class EmotionRecommender:
    def __init__(self):
        """Initializes the model, detects hardware, and sets up label mappings."""
        
        # 1. Detect hardware and the safest model for that hardware
        self.device, self.model_name = self._detect_hardware()

        # 2. Load Model
        print(f"\nLoading AI Model: '{self.model_name}' (This may take a moment)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device
        )
        
        # 3. Label Setup
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

    def _detect_hardware(self):
        """Dynamically finds the best hardware accelerator and matching model."""
        print("Detecting hardware...")
        
        # Define the optimal models
        deberta_model = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        bart_model = "facebook/bart-large-mnli"
        
        # 1. Check for NVIDIA (CUDA) or native Linux AMD (ROCm)
        if torch.cuda.is_available():
            print("Hardware selected: NVIDIA GPU (CUDA)")
            return 0, deberta_model  # 0 targets the first dedicated GPU
            
        # 2. Check for Apple Silicon (M1/M2/M3/M4)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Hardware selected: Apple Silicon (Metal Performance Shaders)")
            return "mps", deberta_model
            
        # 3. Check for Windows AMD (DirectML)
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                print(f"Hardware selected: AMD GPU via DirectML ({device})")
                print("Note: Safely defaulting to BART model to prevent DirectML tensor crashes.")
                return device, bart_model
            except ImportError:
                # 4. Fallback if no GPU or specific libraries are found
                print("Hardware selected: CPU (Warning: Inference will be slower)")
                return -1, deberta_model  # CPU can safely handle DeBERTa, just slowly

    def apply_rules(self, text, scores_dict):
        """Applies keyword-based boosts, clamping the maximum probability to 1.0."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["will", "going to", "about to", "soon", "later", "plan", "expect"]):
            scores_dict["anticipating"] = min(scores_dict.get("anticipating", 0) + 0.15, 1.0)

        if any(word in text_lower for word in ["definitely", "for sure", "guarantee", "reliable", "trust"]):
            scores_dict["trusting"] = min(scores_dict.get("trusting", 0) + 0.15, 1.0)

        if any(word in text_lower for word in ["nervous", "worried", "anxious", "scared", "panic"]):
            scores_dict["fearful"] = min(scores_dict.get("fearful", 0) + 0.20, 1.0)

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

        scores_dict = {
            result['labels'][i]: result['scores'][i]
            for i in range(len(result['labels']))
        }

        scores_dict = self.apply_rules(text, scores_dict)

        valid_emotions = [
            (label, score)
            for label, score in scores_dict.items()
            if score > threshold
        ]

        if not valid_emotions:
            top_label = max(scores_dict, key=scores_dict.get)
            top_score = scores_dict[top_label]
            valid_emotions.append((top_label, top_score))

        valid_emotions.sort(key=lambda x: x[1], reverse=True)
        top_emotions = valid_emotions[:3]

        detected_emotions = {
            self.LABEL_MAP[label]: round(score, 3)
            for label, score in top_emotions
        }

        return detected_emotions

# ==========================================
# TEST WRAPPER
# ==========================================
if __name__ == "__main__":
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

