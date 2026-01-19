import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

# Global variables
tfidf_vectorizer = None
full_df = None

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data():
    global full_df
    try:
        df = pd.read_csv('menu_data.csv')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Combine columns for AI Text Search
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
# 2. PARSERS (Smart Extraction)
# ==========================================
def extract_negations_and_clean(text):
    text = text.lower()
    exclusions = []
    patterns = [
        r'\b(no|not|without|exclude|except|avoid)\s+(\w+)', 
        r'\b(don\'?t|do\s+not)\s+(want|like|need|have)\s+(\w+)'
    ]
    clean_text = text 
    for pat in patterns:
        matches = re.finditer(pat, text)
        for m in matches:
            if m.lastindex:
                banned_word = m.group(m.lastindex)
                exclusions.append(banned_word)
                clean_text = clean_text.replace(m.group(0), " ")
    return exclusions, clean_text

def extract_budget(text):
    text = text.lower()
    MAX_KEYWORDS = ["less than", "under", "below", "cheaper than", "max", "maximum", "up to", "lower than", "budget", "price", "cost"]
    MIN_KEYWORDS = ["more than", "over", "above", "expensive than", "min", "minimum", "at least", "starting from", "higher than"]
    CURRENCY_KEYWORDS = ["pesos", "php", "p", "cost", "price"]
    
    match_min = re.search(r'(' + '|'.join(MIN_KEYWORDS) + r')\s*(\d+)', text)
    if match_min: return ('min', float(match_min.group(2)))
    
    match_max = re.search(r'(' + '|'.join(MAX_KEYWORDS) + r')\s*(\d+)', text)
    if match_max: return ('max', float(match_max.group(2)))
    
    match_curr = re.search(r'(\d+)\s*(' + '|'.join(CURRENCY_KEYWORDS) + r')', text)
    if match_curr: return ('max', float(match_curr.group(1)))
    return None

def extract_category(text):
    text = text.lower()
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
    text = re.sub(r'\b(hot|cold)\s+(outside|weather|day|today)\b', '', text)
    if any(w in text for w in ['cold', 'iced', 'ice', 'frappe', 'frozen', 'chilled', 'cool']): return 'Cold'
    if any(w in text for w in ['hot', 'warm', 'soup', 'stew', 'boiling', 'heating']): return 'Hot'
    return None

def extract_weather(text):
    text = text.lower()
    pattern = r'\b(cold|hot|iced|warm)\s+(drinks?|beverages?|coffees?|teas?|sodas?|waters?|frappes?|meals?|foods?)\b'
    text = re.sub(pattern, '', text)
    weather_keywords = {
        'sunny': 'sunny', 'hot': 'sunny', 'summer': 'sunny',
        'rain': 'rainy', 'raining': 'rainy', 'rainy': 'rainy', 'wet': 'rainy', 'storm': 'rainy',
        'cold': 'cold', 'freezing': 'cold', 'winter': 'cold', 'cloudy': 'cloudy'
    }
    for w in re.findall(r'\b\w+\b', text):
        if w in weather_keywords: return weather_keywords[w]
    return None

def extract_diet(text):
    for d in ['vegan', 'vegetarian', 'pescatarian', 'keto', 'halal']:
        if d in text.lower(): return d.capitalize()
    return None

def extract_mood(text):
    synonyms = {
        'hangry': 'hungry', 'starving': 'hungry', 
        'thirsty': 'refresh', 'refreshing': 'refresh',
        'sad': 'comfort', 'depressed': 'comfort', 'comforting': 'comfort',
        'gym': 'vitality', 'tired': 'vitality', 'energy': 'vitality',
        'happy': 'sweet', 'celebrate': 'sweet'
    }
    words = re.findall(r'\b\w+\b', text.lower())
    clean_words = [synonyms.get(w, w) for w in words if not w.isdigit()]
    return " ".join(clean_words)

# ==========================================
# 3. HELPER CHECKS (ALL PRESENT NOW!)
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
    """Checks if the item matches the specific weather tag."""
    if not user_weather: return True
    return user_weather in str(row_tags).lower()

def check_command_intent(text):
    """
    Checks if the user used a 'Command Word' (e.g., recommend, want, give me).
    """
    command_keywords = [
        'recommend', 'suggest', 'suggestion', 'what should i', 'what do you have',
        'i want', 'i need', 'i like', 'give me', 'show me', 'looking for',
        'can i have', 'can you', 'order', 'menu', 'list', 'crave', 'craving', 'try'
    ]
    text = text.lower()
    return any(keyword in text for keyword in command_keywords)

def parse_user_intent(user_input):
    exclusions, clean_text = extract_negations_and_clean(user_input)
    
    data = {
        'mood_text': extract_mood(clean_text),
        'weather': extract_weather(clean_text),
        'budget': extract_budget(clean_text),
        'diet': extract_diet(clean_text),
        'temperature': extract_temperature(clean_text),
        'category': extract_category(clean_text),
        'exclusions': exclusions
    }
    
    # --- NEW CONTEXT CHECK ---
    # We must strip out "Command Words" to see if there is any REAL context left.
    # Example: "Recommend" -> removes "recommend" -> remains "" -> No Context.
    # Example: "Recommend Coffee" -> removes "recommend" -> remains "Coffee" -> Context!
    
    temp_text = clean_text
    command_keywords = [
        'recommend', 'suggest', 'suggestion', 'what should i', 'what do you have',
        'i want', 'i need', 'i like', 'give me', 'show me', 'looking for',
        'can i have', 'can you', 'order', 'menu', 'list', 'crave', 'craving', 'try'
    ]
    
    for word in command_keywords:
        temp_text = temp_text.replace(word, "")
    
    # Now check if anything useful is left
    has_context = any([
        data['weather'], data['budget'], data['diet'], 
        data['temperature'], data['category'], 
        temp_text.strip() != "" # Check the STRIPPED text, not the raw text
    ])
    
    return data, has_context

def get_fallback_recommendations():
    return full_df.sort_values('price').head(5).to_dict('records')

# ==========================================
# 4. RECOMMENDATION ENGINE
# ==========================================

def recommend(parse_data):
    user_tfidf = tfidf_vectorizer.transform([parse_data['mood_text']])
    
    active_criteria = []
    if parse_data['category']: active_criteria.append('Category')
    if parse_data['temperature']: active_criteria.append('Temperature')
    if parse_data['weather']: active_criteria.append('Weather')
    
    known_moods = ['hungry', 'refresh', 'comfort', 'vitality', 'sweet', 'savory']
    user_moods = [w for w in parse_data['mood_text'].split() if w in known_moods]
    if user_moods: active_criteria.append('Mood')

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
        
        # Category Check
        if 'Category' in active_criteria:
            if not check_category_hit(row, parse_data['category']):
                fails += 1; missed_tags.append('Category')
            else: hits += 1

        # Temperature Check
        if 'Temperature' in active_criteria:
            if str(row['temperature']).lower() != parse_data['temperature'].lower():
                fails += 1; missed_tags.append('Temperature')
            else: hits += 1

        # Weather Check (Using Helper Function!)
        if 'Weather' in active_criteria:
            weather_hit = check_weather_hit(row['weather_tags'], parse_data['weather'])
            
            # Comfort Logic (Cold day -> Hot food)
            comfort_hit = False
            item_temp = str(row['temperature']).lower()
            w = parse_data['weather']
            if w in ['cold', 'rainy'] and 'hot' in item_temp: comfort_hit = True
            elif w in ['sunny', 'hot'] and 'cold' in item_temp: comfort_hit = True
            
            if weather_hit or comfort_hit: hits += 1
            else: fails += 1; missed_tags.append('Weather')

        # Mood Check
        if 'Mood' in active_criteria:
            item_moods = str(row['mood_tags']).lower()
            if any(m in item_moods for m in user_moods): hits += 1
            else: fails += 1; missed_tags.append('Mood')

        if fails >= 2: continue
            
        total_possible = len(active_criteria)
        if total_possible == 0: percentage = 100
        else: percentage = int((hits / total_possible) * 100)

        item_vector = tfidf_vectorizer.transform([full_df.iloc[index]['features']])
        text_score = linear_kernel(user_tfidf, item_vector)[0][0]
        
        candidates.append({
            'name': row['name'], 'price': row['price'],
            'percentage': percentage, 'text_score': text_score, 'missed': missed_tags
        })
            
    if not candidates and (active_criteria or parse_data['budget'] or parse_data['diet']):
        print("   ⚠️  Criteria too strict. Showing Chef's Recommendations:")
        fallback = get_fallback_recommendations()
        for item in fallback:
            item['percentage'] = 0; item['text_score'] = 0; item['missed'] = ['Fallback']
        return fallback

    candidates = sorted(candidates, key=lambda x: (-x['percentage'], -x['text_score'], x['price']))
    return candidates[:5]

if __name__ == "__main__":
    if load_data() is not None:
        build_model(full_df)
        print("\n🤖 STRICT BOT: Ready! (Requires Command AND Context)")
        
        while True:
            u = input("\nYou: ")
            if u.lower() in ['q', 'quit']: break
            if not u.strip(): continue
            
            p, has_context = parse_user_intent(u)
            has_command = check_command_intent(u)
            
            if has_context and not has_command:
                print("   ℹ️  I noticed the context. Do you want me to recommend something?")
                continue 

            if not has_context and has_command:
                print("   ❓ I don't know what recommendations you want, can you clarify it? (e.g., 'I want coffee')")
                continue 

            if not has_context and not has_command:
                print("   👋 Hi there! I can help you find food. Try saying 'I want a cold drink'.")
                continue

            recs = recommend(p)
            
            if not recs:
                print("   ⚠️  No items found.")
            else:
                for item in recs:
                    score = item.get('percentage', 0)
                    missed = ", ".join(item.get('missed', [])) if item.get('missed') else "None"
                    
                    if missed == 'Fallback':
                        print(f"   💡 [Chef's Rec] {item['name']} (₱{item['price']})")
                    elif score == 100:
                        print(f"   ★ {item['name']} (₱{item['price']}) - 100% Match!")
                    else:
                        print(f"   ☆ {item['name']} (₱{item['price']}) - {score}% (Missed: {missed})")