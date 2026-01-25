import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 0. GLOBAL CONFIGURATION & BRAIN LOADING
# ==========================================
print("   🧠 Loading AI Brain (this may take a moment)...")
# We use a small, fast model optimized for laptops
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

tfidf_vectorizer = None
full_df = None

# These are the "Concepts" your menu understands. 
# The AI will map user input to the closest concept here.
TARGET_MOODS = [
    'vitality', 'celebratory', 'cozy', 'comfort', 'stressed', 
    'hangry', 'sick', 'romantic', 'curiosity', 'routine', 'refresh'
]

# Pre-calculate the mathematical positions of your moods
mood_embeddings = semantic_model.encode(TARGET_MOODS, convert_to_tensor=True)

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data():
    global full_df
    try:
        df = pd.read_csv('menu_data.csv')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Combine columns for AI Text Search (Keyword matching layer)
        df['features'] = (
            df['name'] + " " + 
            df['ingredients'] + " " + 
            df['mood_tags'] + " " + 
            df['meal_type'] + " " +
            df['weather_tags'].fillna('') + " " +
            df['temperature'].fillna('') + " " + 
            df['main_category'].fillna('') + " " +
            df['sub_category'].fillna('') + " " +
            df['base'].fillna('')
        )
        df['features'] = df['features'].fillna('')
        full_df = df
        return df
    except FileNotFoundError:
        print("❌ Error: 'menu_data.csv' not found.")
        return None

def build_model(df):
    global tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['features'])

# ==========================================
# 2. PARSERS (Hybrid: Regex + Semantic)
# ==========================================

def extract_negations_and_clean(text):
    """
    Removes negative constraints like "not spicy" so the AI doesn't match "spicy".
    Returns a list of excluded words.
    """
    text = text.lower()
    exclusions = []
    # Patterns to catch "not spicy", "isn't too sweet", "avoid pork", etc.
    patterns = [
        r'\b(no|not|isn\'?t|aren\'?t|less|without|exclude|except|avoid)\s+(too\s+|so\s+|very\s+)?(\w+)', 
        r'\b(don\'?t|do\s+not)\s+(want|like|need|have)\s+(\w+)'
    ]
    clean_text = text 
    for pat in patterns:
        matches = re.finditer(pat, text)
        for m in matches:
            # The banned word is in the last capture group
            # m.groups() returns all groups; we grab the last one which is the noun/adjective
            groups = m.groups()
            banned_word = groups[-1] 
            exclusions.append(banned_word)
            clean_text = clean_text.replace(m.group(0), " ")
            
    return exclusions, clean_text

def extract_mood_semantic(text):
    """
    Uses the AI Brain to find the "vibe" of the input.
    """
    if not text.strip(): return ""
    
    # 1. Convert user text to numbers
    user_embedding = semantic_model.encode(text, convert_to_tensor=True)
    
    # 2. Compare against our target moods
    scores = util.cos_sim(user_embedding, mood_embeddings)[0]
    
    # 3. Find the best match
    best_score_index = scores.argmax()
    best_score = float(scores[best_score_index])
    
    # 4. Confidence Threshold (0.3 avoids matching garbage text)
    if best_score < 0.3:
        return ""
        
    detected_mood = TARGET_MOODS[best_score_index]
    # Debug print to show "Thinking" process
    # print(f"DEBUG: '{text}' matched '{detected_mood}' ({best_score:.2f})")
    return detected_mood

def extract_budget(text):
    text = text.lower()
    MAX_KEYWORDS = ["less than", "under", "below", "cheaper than", "max", "maximum", "up to", "lower than", "budget"]
    MIN_KEYWORDS = ["more than", "over", "above", "expensive than", "min", "minimum", "at least", "starting from"]
    CURRENCY_KEYWORDS = ["pesos", "php", "p"]
    
    match_min = re.search(r'(' + '|'.join(MIN_KEYWORDS) + r')\s*(\d+)', text)
    if match_min: return ('min', float(match_min.group(2)))
    
    match_max = re.search(r'(' + '|'.join(MAX_KEYWORDS) + r')\s*(\d+)', text)
    if match_max: return ('max', float(match_max.group(2)))
    
    match_curr = re.search(r'(\d+)\s*(' + '|'.join(CURRENCY_KEYWORDS) + r')', text)
    if match_curr: return ('max', float(match_curr.group(1)))
    return None

def extract_category(text):
    text = text.lower()
    # Regex is still better for strict categories than semantic search
    categories = {
        'non coffee': 'Non-Coffee', 'non-coffee': 'Non-Coffee',
        'milk tea': 'Milk Tea', 'fruit tea': 'Fruit Tea', 'sparkling soda': 'Sparkling Soda',
        'coffee': 'Coffee', 'tea': 'Tea', 'soda': 'Sparkling Soda', 'frappe': 'Frappe',
        'waffle': 'Waffle', 'pasta': 'Pasta', 'bread': 'Bread',
        'food': 'Food', 'meal': 'Meal', 'lunch': 'Meal', 'dinner': 'Meal',
        'drink': 'Beverage', 'beverage': 'Beverage', 'snack': 'Snack', 'dessert': 'Snack'
    }
    sorted_keys = sorted(categories.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in text: return categories[key]
    return None

def extract_temperature(text):
    text = text.lower()
    # Explicit "Hot" or "Cold" triggers
    if any(w in text for w in ['cold', 'iced', 'ice', 'frappe', 'frozen', 'chilled', 'cool']): return 'Cold'
    if any(w in text for w in ['hot', 'warm', 'soup', 'stew', 'boiling', 'heating']): return 'Hot'
    
    # Metaphor Rule: "Hug" implies Warmth
    if 'hug' in text: return 'Hot'
    return None

def extract_weather(text):
    text = text.lower()
    weather_keywords = {
        'sunny': 'sunny', 'hot': 'sunny', 'summer': 'sunny',
        'rain': 'rainy', 'raining': 'rainy', 'wet': 'rainy', 'storm': 'rainy',
        'cold': 'cold', 'freezing': 'cold', 'winter': 'cold', 'cloudy': 'cloudy'
    }
    for w in re.findall(r'\b\w+\b', text):
        if w in weather_keywords: return weather_keywords[w]
    return None

def extract_diet(text):
    for d in ['vegan', 'vegetarian', 'pescatarian', 'keto', 'halal']:
        if d in text.lower(): return d.capitalize()
    return None

# ==========================================
# 3. HELPER CHECKS & INTENT LOGIC
# ==========================================

def check_budget_hit(row_price, budget_constraint):
    if not budget_constraint: return True
    operator, value = budget_constraint
    if operator == 'max': return row_price <= value
    if operator == 'min': return row_price >= value
    return True

def check_diet_hit(row_tags, user_diet):
    if not user_diet: return True
    tags = str(row_tags)
    if f"Non-{user_diet}" in tags: return False
    return user_diet in tags

def check_category_hit(row, user_cat):
    if not user_cat: return True
    blob = f"{row['main_category']} {row['sub_category']} {row['meal_type']} {row['base']}".lower()
    search = user_cat.lower()
    if "non-" + search in blob and "non-" not in search: return False
    return search in blob

def check_weather_hit(row_tags, user_weather):
    if not user_weather: return True
    return user_weather in str(row_tags).lower()

def check_command_intent(text):
    command_keywords = [
        'recommend', 'suggest', 'suggestion', 'what should i', 'what do you have',
        'i want', 'i need', 'i like', 'give me', 'show me', 'looking for',
        'can i have', 'can you', 'order', 'menu', 'list', 'crave', 'craving', 'try',
        'something', 'anything', 'whatever', 'surprise me'
    ]
    text = text.lower()
    return any(keyword in text for keyword in command_keywords)

def parse_user_intent(user_input):
    # 1. Remove "not X" phrases first
    exclusions, clean_text = extract_negations_and_clean(user_input)
    
    # 2. Extract specific entities using Regex
    data = {
        'weather': extract_weather(clean_text),
        'budget': extract_budget(clean_text),
        'diet': extract_diet(clean_text),
        'temperature': extract_temperature(clean_text),
        'category': extract_category(clean_text),
        'exclusions': exclusions,
        'mood_text': '' # Placeholder
    }

    # 3. Remove detected entities from text to leave "just the mood" for the brain
    # (This helps the brain focus on the emotion, not the keywords)
    temp_text = clean_text
    if data['weather']: temp_text = temp_text.replace(data['weather'], "")
    if data['category']: temp_text = temp_text.replace(data['category'].lower(), "")
    
    # 4. Use the Semantic Brain for Mood
    data['mood_text'] = extract_mood_semantic(temp_text)
    
    # 5. Context Check
    # Strip command words to see if there is any real content left
    check_text = clean_text
    command_keywords = [
        'recommend', 'suggest', 'suggestion', 'what should i', 'what do you have',
        'i want', 'i need', 'i like', 'give me', 'show me', 'looking for',
        'can i have', 'can you', 'order', 'menu', 'list', 'crave', 'craving', 'try',
        'something', 'anything', 'whatever', 'surprise me'
    ]
    for word in command_keywords:
        check_text = check_text.replace(word, "")
    
    has_context = any([
        data['weather'], data['budget'], data['diet'], 
        data['temperature'], data['category'], data['mood_text'],
        check_text.strip() != "" 
    ])
    
    return data, has_context

def get_fallback_recommendations():
    recs = full_df.sort_values('price').head(10).to_dict('records')
    for item in recs:
        item['percentage'] = 0 
        item['text_score'] = 0
        item['missed'] = ['Fallback Recommendation']
    return recs

def print_analysis(p):
    print("\n   🧠 AI Analysis:")
    hit = False
    if p['weather']: print(f"      • Weather: '{p['weather']}'"); hit = True
    if p['mood_text']: print(f"      • Mood (Semantic): '{p['mood_text']}'"); hit = True
    if p['category']: print(f"      • Category: '{p['category']}'"); hit = True
    if p['temperature']: print(f"      • Temp: '{p['temperature']}'"); hit = True
    if p['budget']: print(f"      • Budget: {p['budget'][0]} {p['budget'][1]}"); hit = True
    if p['exclusions']: print(f"      • Exclusions: {p['exclusions']}"); hit = True
    if not hit: print("      • No specific tags (General Search)")
    print("   --------------------------------")

# ==========================================
# 4. RECOMMENDATION ENGINE
# ==========================================

def recommend(parse_data):
    user_tfidf = tfidf_vectorizer.transform([parse_data['mood_text']])
    
    active_criteria = []
    if parse_data['category']: active_criteria.append('Category')
    if parse_data['temperature']: active_criteria.append('Temperature')
    if parse_data['weather']: active_criteria.append('Weather')
    if parse_data['mood_text']: active_criteria.append('Mood') # Now semantic!

    candidates = []
    
    for index, row in full_df.iterrows():
        # --- STRICT FILTERS ---
        if parse_data['exclusions']:
            if any(banned in row['features'].lower() for banned in parse_data['exclusions']): continue
        if parse_data['budget'] and not check_budget_hit(row['price'], parse_data['budget']): continue 
        if parse_data['diet'] and not check_diet_hit(row['dietary_tags'], parse_data['diet']): continue
        
        # --- SOFT SCORING ---
        fails = 0
        hits = 0
        missed_tags = []
        
        if 'Category' in active_criteria:
            if not check_category_hit(row, parse_data['category']):
                fails += 1; missed_tags.append('Category')
            else: hits += 1

        if 'Temperature' in active_criteria:
            if str(row['temperature']).lower() != parse_data['temperature'].lower():
                fails += 1; missed_tags.append('Temperature')
            else: hits += 1

        if 'Weather' in active_criteria:
            weather_hit = check_weather_hit(row['weather_tags'], parse_data['weather'])
            comfort_hit = False
            item_temp = str(row['temperature']).lower()
            w = parse_data['weather']
            # Smart logic: Cold drink for Hot weather
            if w in ['cold', 'rainy'] and 'hot' in item_temp: comfort_hit = True
            elif w in ['sunny', 'hot'] and 'cold' in item_temp: comfort_hit = True
            
            if weather_hit or comfort_hit: hits += 1
            else: fails += 1; missed_tags.append('Weather')

        if 'Mood' in active_criteria:
            item_moods = str(row['mood_tags']).lower()
            # Since semantic match gives us one specific mood (e.g. 'stressed'),
            # we check if that specific mood tag exists in the item.
            if parse_data['mood_text'] in item_moods: hits += 1
            else: fails += 1; missed_tags.append('Mood')

        # Drop item if it fails 2 or more active criteria
        if fails >= 2: continue
            
        total_possible = len(active_criteria)
        if total_possible == 0: percentage = 100
        else: percentage = int((hits / total_possible) * 100)

        # Calculate secondary text score for sorting similar items
        item_vector = tfidf_vectorizer.transform([full_df.iloc[index]['features']])
        text_score = linear_kernel(user_tfidf, item_vector)[0][0]
        
        candidates.append({
            'name': row['name'], 'price': row['price'],
            'percentage': percentage, 'text_score': text_score, 'missed': missed_tags
        })

    # --- FILTER LOGIC (TOP TIER ONLY) ---
    valid_candidates = [c for c in candidates if c['percentage'] > 0]
    
    if valid_candidates:
        # Find the highest score (e.g., 100) and show only those
        best_score = max(c['percentage'] for c in valid_candidates)
        candidates = [c for c in valid_candidates if c['percentage'] == best_score]
    else:
        candidates = [] 

    if not candidates and (active_criteria or parse_data['budget'] or parse_data['diet']):
        print("   ⚠️  No exact matches found. Showing Chef's Recommendations:")
        return get_fallback_recommendations()

    candidates = sorted(candidates, key=lambda x: (-x['percentage'], -x['text_score'], x['price']))
    
    return candidates 

if __name__ == "__main__":
    if load_data() is not None:
        build_model(full_df)
        print("\n🤖 SMART BOT: Ready! (Semantic Brain Active)")
        
        while True:
            u = input("\nYou: ")
            if u.lower() in ['q', 'quit']: break
            if not u.strip(): continue
            
            p, has_context = parse_user_intent(u)
            has_command = check_command_intent(u)
            
            # --- INTERACTION LOGIC ---
            
            if has_context and not has_command:
                print_analysis(p)
                print("   ℹ️  I noticed these preferences. Do you want me to recommend something?")
                continue 

            if not has_context and has_command:
                wildcards = ['something', 'anything', 'whatever', 'surprise me']
                if any(w in u.lower() for w in wildcards):
                    print("   🎲 You're feeling adventurous! Here are our best sellers:")
                    recs = get_fallback_recommendations()
                    for item in recs:
                        print(f"   ★ {item['name']} (₱{item['price']})")
                    continue
                else:
                    print("   ❓ I don't know what recommendations you want, can you clarify it?")
                    continue 

            if not has_context and not has_command:
                print("   👋 Hi there! Try saying 'I want a cold drink' or 'I had a bad day'.")
                continue

            print_analysis(p)
            recs = recommend(p)
            
            if not recs:
                print("   ⚠️  No items found.")
            else:
                for item in recs:
                    score = item.get('percentage', 0)
                    missed = ", ".join(item.get('missed', [])) if item.get('missed') else "None"
                    
                    if missed == 'Fallback Recommendation':
                        print(f"   💡 [Chef's Rec] {item['name']} (₱{item['price']})")
                    elif score == 100:
                        print(f"   ★ {item['name']} (₱{item['price']}) - 100% Match!")
                    else:
                        print(f"   ☆ {item['name']} (₱{item['price']}) - {score}% (Missed: {missed})")