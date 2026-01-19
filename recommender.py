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
        # --- SOLAR (Hot/Sunny) ---
        'sunny': 'solar', 'sun': 'solar', 'shine': 'solar',
        'hot': 'solar', 'heat': 'solar', 'heatwave': 'solar',
        'scorching': 'solar', 'boiling': 'solar', 'burning': 'solar',
        'summer': 'solar', 'dry': 'solar', 'arid': 'solar',
        'warm': 'solar', 'humid': 'solar', 'muggy': 'solar',
        'sweat': 'solar', 'sweating': 'solar', 'thirsty': 'solar',
        'bright': 'solar', 'clear': 'solar', 'beach': 'solar',

        # --- POLAR (Cold/Freezing) ---
        'cold': 'polar', 'cool': 'polar', 'chilly': 'polar',
        'freezing': 'polar', 'freeze': 'polar', 'frozen': 'polar',
        'winter': 'polar', 'snow': 'polar', 'snowy': 'polar',
        'ice': 'polar', 'icy': 'polar', 'frost': 'polar',
        'blizzard': 'polar', 'shiver': 'polar', 'shivering': 'polar',
        'jacket': 'polar', 'coat': 'polar', 'scarf': 'polar',

        # --- HYDRO (Rain/Wet) ---
        'rain': 'hydro', 'rainy': 'hydro', 'raining': 'hydro',
        'wet': 'hydro', 'water': 'hydro', 'soak': 'hydro',
        'storm': 'hydro', 'stormy': 'hydro', 'thunder': 'hydro',
        'lightning': 'hydro', 'typhoon': 'hydro', 'hurricane': 'hydro',
        'monsoon': 'hydro', 'flood': 'hydro', 'pouring': 'hydro',
        'drizzle': 'hydro', 'shower': 'hydro', 'umbrella': 'hydro',

        # --- MILD (Neutral/Cloudy) ---
        'cloudy': 'mild', 'clouds': 'mild', 'overcast': 'mild',
        'gray': 'mild', 'grey': 'mild', 'gloomy': 'mild',
        'windy': 'mild', 'wind': 'mild', 'breeze': 'mild',
        'fine': 'mild', 'nice': 'mild', 'okay': 'mild',
        'fresh': 'mild', 'fair': 'mild', 'dull': 'mild'
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
        # --- BRANCH 1: VITALITY (Healthy/Focus) ---
        'gym': 'vitality', 'workout': 'vitality', 'healthy': 'vitality',
        'diet': 'vitality', 'fresh': 'vitality', 'light': 'vitality',
        'focus': 'vitality', 'study': 'vitality', 'productive': 'vitality',
        'morning': 'vitality', 'breakfast': 'vitality', 'energetic': 'vitality',
        'clean': 'vitality', 'fit': 'vitality',
        
        # --- BRANCH 2: CELEBRATORY (Party/Fun) ---
        'party': 'celebratory', 'friends': 'celebratory', 'birthday': 'celebratory',
        'winning': 'celebratory', 'promotion': 'celebratory', 'awesome': 'celebratory',
        'good': 'celebratory', 'happy': 'celebratory', 'great': 'celebratory',
        'yay': 'celebratory', 'treat': 'celebratory', 'share': 'celebratory',
        'fun': 'celebratory', 'group': 'celebratory',

        # --- BRANCH 3: CHILL (Relaxed/Cozy) ---
        'chill': 'cozy', 'relax': 'cozy', 'rainy': 'cozy',
        'cold': 'cozy', 'book': 'cozy', 'reading': 'cozy',
        'movie': 'cozy', 'netflix': 'cozy', 'calm': 'cozy',
        'quiet': 'cozy', 'afternoon': 'cozy', 'lazy': 'cozy',
        'warm': 'cozy',

        # --- BRANCH 4: COMFORT (Sad/Emotional) ---
        'sad': 'comfort', 'crying': 'comfort', 'depressed': 'comfort',
        'lonely': 'comfort', 'bad': 'comfort', 'terrible': 'comfort',
        'heartbroken': 'comfort', 'miss': 'comfort', 'grief': 'comfort',
        'emotional': 'comfort', 'blue': 'comfort', 'hard': 'comfort',

        # --- BRANCH 5: STRESSED (Angry/Rushed) ---
        'stressed': 'stressed', 'busy': 'stressed', 'deadline': 'stressed',
        'rushed': 'stressed', 'panic': 'stressed', 'angry': 'stressed',
        'furious': 'stressed', 'annoyed': 'stressed', 'mad': 'stressed',
        'late': 'stressed', 'crunchy': 'stressed', 'fast': 'stressed',

        # --- BRANCH 6: HANGRY (Starving/Craving) ---
        'hungry': 'hangry', 'starving': 'hangry', 'famished': 'hangry',
        'heavy': 'hangry', 'greasy': 'hangry', 'salty': 'hangry', 'huge': 'hangry',
        'feast': 'hangry', 'full': 'hangry', 'big': 'hangry',

        # --- BRANCH 7: SICK (Unwell) ---
        'sick': 'sick', 'flu': 'sick', 'ill': 'sick',
        'headache': 'sick', 'stomachache': 'sick', 'pain': 'sick',
        'hangover': 'sick', 'drunk': 'sick', 'tired': 'sick',
        'exhausted': 'sick', 'nausea': 'sick', 'unwell': 'sick',

        # --- BRANCH 8: ROMANTIC (Date Night) ---
        'love': 'romantic', 'anniversary': 'romantic',
        'fancy': 'romantic', 'couple': 'romantic',
        'special': 'romantic', 'candle': 'romantic'
    }
    words = re.findall(r'\b\w+\b', text.lower())
    clean_words = [synonyms.get(w, w) for w in words if not w.isdigit()]
    return " ".join(clean_words)

# ==========================================
# 3. NEW: THE GATEKEEPER LOGIC
# ==========================================

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
    
    # Check if ANY context was found (is the dictionary empty of useful data?)
    has_context = any([
        data['weather'], data['budget'], data['diet'], 
        data['temperature'], data['category'], 
        data['mood_text'].strip() != ""
    ])
    
    return data, has_context

# ==========================================
# 4. RECOMMENDATION ENGINE
# ==========================================
# (Standard helper functions omitted for brevity, they are same as before)
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

def check_diet_hit(row_tags, user_diet):
    if not user_diet: return True
    tags = str(row_tags)
    if f"Non-{user_diet}" in tags: return False
    return user_diet in tags

def get_fallback_recommendations():
    return full_df.sort_values('price').head(5).to_dict('records')

def recommend(parse_data):
    user_tfidf = tfidf_vectorizer.transform([parse_data['mood_text']])
    
    # 1. Identify which Criteria are ACTIVE
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
        
        if 'Category' in active_criteria:
            blob = f"{row['main_category']} {row['sub_category']} {row['meal_type']} {row['base']}".lower()
            if parse_data['category'].lower() not in blob:
                fails += 1; missed_tags.append('Category')
            else: hits += 1

        if 'Temperature' in active_criteria:
            if str(row['temperature']).lower() != parse_data['temperature'].lower():
                fails += 1; missed_tags.append('Temperature')
            else: hits += 1

        if 'Weather' in active_criteria:
            weather_hit = parse_data['weather'] in str(row['weather_tags']).lower()
            comfort_hit = False
            item_temp = str(row['temperature']).lower()
            w = parse_data['weather']
            if w in ['cold', 'rainy'] and 'hot' in item_temp: comfort_hit = True
            elif w in ['sunny', 'hot'] and 'cold' in item_temp: comfort_hit = True
            
            if weather_hit or comfort_hit: hits += 1
            else: fails += 1; missed_tags.append('Weather')

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
            
    # Fallback if specific search yielded zero results
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
            
            # 1. Parse Data and Check for Content
            p, has_context = parse_user_intent(u)
            
            # 2. Check for Command Words ("recommend", "want", etc.)
            has_command = check_command_intent(u)
            
            # ============================================
            # GATEKEEPER LOGIC (Your Requested Rules)
            # ============================================
            
            # Case A: Context YES, Command NO (e.g. "It's hot outside")
            if has_context and not has_command:
                print("   ℹ️  I noticed the context. Do you want me to recommend something?")
                continue # Do NOT run recommender

            # Case B: Context NO, Command YES (e.g. "Recommend me something")
            if not has_context and has_command:
                print("   ❓ I don't know what recommendations you want, can you clarify it? (e.g., 'I want coffee')")
                continue # Do NOT run recommender

            # Case C: Context NO, Command NO (e.g. "Hello", "test")
            if not has_context and not has_command:
                print("   👋 Hi there! I can help you find food. Try saying 'I want a cold drink'.")
                continue

            # Case D: BOTH PRESENT -> RUN ENGINE
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