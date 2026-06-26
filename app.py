"""
DishRadar AI — Restaurant Marketing Intelligence Platform
Reviews + competitor intel → ready-to-post visual marketing content
"""

import base64
import inspect
import json
import os
import re
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
analyzer = SentimentIntensityAnalyzer()

# Gemini client for Nano Banana image generation
try:
    from google import genai as google_genai
    _gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_client = google_genai.Client(api_key=_gemini_key)
    GEMINI_AVAILABLE = bool(_gemini_key)
except ImportError as _ie:
    gemini_client    = None
    GEMINI_AVAILABLE = False
    _GEMINI_ERR      = f"google-genai not installed: {_ie}"
except Exception as _ee:
    gemini_client    = None
    GEMINI_AVAILABLE = False
    _GEMINI_ERR      = f"Gemini init error: {_ee}"
else:
    _GEMINI_ERR = None

try:
    from apify_client import ApifyClient
    apify_client   = ApifyClient(os.getenv("APIFY_API_TOKEN", ""))
    APIFY_AVAILABLE = bool(os.getenv("APIFY_API_TOKEN"))
except ImportError:
    apify_client   = None
    APIFY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DishRadar AI", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:         #080a0e;
    --surface:    #0f1218;
    --raised:     #161b24;
    --border:     #1e2535;
    --amber:      #e8870a;
    --amber-dim:  rgba(232,135,10,0.12);
    --parchment:  #f2ead8;
    --slate:      #8a9bb0;
    --pos:        #3ec99a;
    --neg:        #e05c6a;
    --comp:       #7b8ff7;
    --text:       #dde3ed;
    --text-soft:  #a0adbf;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
  }
  [data-testid="stSidebar"]      { background: var(--surface) !important; }
  [data-testid="stHeader"]       { background: transparent !important; }
  [data-testid="stToolbar"]      { display: none !important; }

  /* ── Masthead ── */
  .masthead {
    padding: 3.5rem 0 3rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3rem;
  }
  .masthead-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.75rem;
  }
  .masthead-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.6rem;
    font-weight: 400;
    letter-spacing: -0.5px;
    color: var(--parchment);
    line-height: 1;
    margin: 0 0 0.6rem;
  }
  .masthead-title em { color: var(--amber); font-style: normal; }
  .masthead-sub {
    font-size: 0.9rem;
    color: var(--slate);
    font-weight: 300;
    letter-spacing: 0.01em;
  }

  /* ── Input area ── */
  .input-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.6rem;
    display: block;
  }
  .input-label.comp { color: var(--comp); }

  div[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.15s;
  }
  div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--amber) !important;
    outline: none !important;
  }

  div[data-testid="stButton"] > button {
    background: var(--amber) !important;
    color: #080a0e !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.15s !important;
  }
  div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

  /* ── Section divider ── */
  .sec-head {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1.25rem;
  }
  .sec-head-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber);
    white-space: nowrap;
  }
  .sec-head-label.comp { color: var(--comp); }
  .sec-head-label.pos  { color: var(--pos); }
  .sec-head-rule {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Stat strip ── */
  .stat-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 2.5rem;
  }
  .stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.1rem 0.9rem;
  }
  .stat-box .sl {
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 0.3rem;
  }
  .stat-box .sv {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
  }
  .stat-box.tot .sv { color: var(--parchment); }
  .stat-box.pos .sv { color: var(--pos); }
  .stat-box.neg .sv { color: var(--neg); }
  .stat-box.neu .sv { color: var(--slate); }
  .stat-box.opp .sv { color: var(--comp); }

  /* ── Pill tags ── */
  .pill-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.8rem; }
  .pill {
    border-radius: 4px;
    padding: 0.2rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.03em;
  }
  .pill.pos { background: rgba(62,201,154,0.1); color: var(--pos); border: 1px solid rgba(62,201,154,0.25); }
  .pill.neg { background: rgba(224,92,106,0.1); color: var(--neg); border: 1px solid rgba(224,92,106,0.25); }
  .pill.opp { background: rgba(123,143,247,0.1); color: var(--comp); border: 1px solid rgba(123,143,247,0.25); }

  /* ── Quote blocks ── */
  .qline {
    border-left: 2px solid var(--border);
    padding: 0.45rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.84rem;
    color: var(--text-soft);
    line-height: 1.55;
    font-style: italic;
  }
  .qline.pos { border-left-color: var(--pos); color: var(--text); font-style: normal; }
  .qline.neg { border-left-color: var(--neg); }
  .qline.opp { border-left-color: var(--comp); }

  /* ── Content cards ── */
  .ccard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.15s;
  }
  .ccard:hover { border-color: var(--amber); }
  .ccard .cl {
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.55rem;
  }
  .ccard.comp-c .cl { color: var(--comp); }
  .ccard.promo-c .cl { color: #c9a227; }
  .ccard .cb {
    font-size: 0.93rem;
    color: var(--text);
    line-height: 1.65;
  }
  .ccard .cr {
    font-size: 0.76rem;
    color: var(--slate);
    margin-top: 0.55rem;
  }

  /* ── Poster card ── */
  .poster-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.8rem;
    text-align: center;
  }
  .poster-wrap img {
    width: 100%;
    max-width: 360px;
    border-radius: 8px;
    display: block;
    margin: 0 auto 1rem;
  }
  .poster-caption {
    font-size: 0.86rem;
    color: var(--text);
    line-height: 1.6;
    text-align: left;
    margin-bottom: 0.5rem;
  }
  .poster-note {
    font-size: 0.72rem;
    color: var(--slate);
    text-align: left;
  }

  /* ── Action list ── */
  .action-row {
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    padding: 0.9rem 1.2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.6rem;
  }
  .action-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--amber);
    line-height: 1;
    min-width: 24px;
    padding-top: 0.05rem;
  }
  .action-txt {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.55;
    padding-top: 0.15rem;
  }

  hr { border-color: var(--border) !important; }
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  [data-testid="stExpander"] { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-eyebrow">Restaurant Marketing Intelligence</div>
  <h1 class="masthead-title">Dish<em>Radar</em></h1>
  <p class="masthead-sub">Turn customer reviews and competitor weaknesses into ready-to-post content.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown('<span class="input-label">Your Restaurant</span>', unsafe_allow_html=True)
    restaurant_name     = st.text_input("rname",     placeholder="Restaurant name",   label_visibility="collapsed")
    restaurant_location = st.text_input("rlocation", placeholder="City, State",       label_visibility="collapsed")

with col_b:
    st.markdown('<span class="input-label comp">Competitor — optional</span>', unsafe_allow_html=True)
    competitor_name     = st.text_input("cname",     placeholder="Competitor name",   label_visibility="collapsed")
    competitor_location = st.text_input("clocation", placeholder="City, State",       label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
num_posts = st.radio(
    "Instagram posts to generate",
    options=[1, 2, 3],
    index=2,
    horizontal=True,
)
run_btn = st.button("Generate content →")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
THEME_KEYWORDS = {
    "Food Quality": ["food", "taste", "flavor", "fresh", "delicious", "cold", "stale", "bland"],
    "Service":      ["service", "staff", "waiter", "server", "rude", "friendly", "helpful"],
    "Wait Time":    ["wait", "waiting", "slow", "delayed", "forever", "late"],
    "Price":        ["price", "expensive", "cheap", "overpriced", "value"],
    "Portion Size": ["portion", "small", "large", "quantity", "serving"],
    "Atmosphere":   ["atmosphere", "ambience", "vibe", "music", "decor"],
    "Cleanliness":  ["clean", "dirty", "bathroom", "hygiene"],
    "Parking":      ["parking", "valet", "park"],
}
THEME_PATTERNS = {
    theme: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for theme, kws in THEME_KEYWORDS.items()
}


def get_sentiment(text):
    s = analyzer.polarity_scores(str(text))["compound"]
    return (s, "Positive") if s >= 0.05 else (s, "Negative") if s <= -0.05 else (s, "Neutral")


def build_sentiment_df(texts, source):
    rows = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        score, label = get_sentiment(t)
        rows.append({"source": source, "text": t, "sentiment_score": score, "sentiment_label": label})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","text","sentiment_score","sentiment_label"])


def extract_themes(df, sentiment_filter):
    if df.empty or "sentiment_label" not in df.columns:
        return {}
    subset = df[df["sentiment_label"] == sentiment_filter]["text"].astype(str).tolist()
    return {
        theme: [t for t in subset if pattern.search(t)]
        for theme, pattern in THEME_PATTERNS.items()
        if any(pattern.search(t) for t in subset)
    }


def fetch_place_data(name, location, max_reviews=50):
    """
    Fetch full place data from Apify — reviews AND brand profile fields.
    Returns (reviews: list[str], brand_profile: dict)
    """
    if not APIFY_AVAILABLE or apify_client is None:
        return [], {}

    query = f"{name} {location}".strip()
    run_input = {
        "searchStringsArray": [query],
        "maxCrawledPlacesPerSearch": 1,
        "maxReviews": max_reviews,
        "language": "en",
        "reviewsSort": "newest",
    }
    try:
        run    = apify_client.actor("compass/crawler-google-places").call(run_input=run_input)
        reviews       = []
        brand_profile = {}

        for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
            # Skip if item is not a dict (defensive)
            if not isinstance(item, dict):
                continue

            # ── Brand profile — pulled once from the first result ──
            if not brand_profile:
                try:
                    # Photo URLs — handle both list-of-strings and list-of-dicts
                    raw_photos = item.get("imageUrls", []) or []
                    if not raw_photos:
                        for img in item.get("images", []):
                            if isinstance(img, dict) and img.get("imageUrl"):
                                raw_photos.append(img["imageUrl"])
                            elif isinstance(img, str) and img.startswith("http"):
                                raw_photos.append(img)
                    photo_urls = [p for p in raw_photos if isinstance(p, str) and p.startswith("http")][:6]

                    # Menu highlights — handle if menu is not a dict
                    menu = item.get("menu") or {}
                    menu_highlights = []
                    if isinstance(menu, dict):
                        for h in menu.get("highlights", []):
                            if isinstance(h, dict):
                                menu_highlights.append(h.get("title", ""))
                            elif isinstance(h, str):
                                menu_highlights.append(h)

                    brand_profile = {
                        "name":            item.get("title", name),
                        "category":        item.get("categoryName", ""),
                        "categories":      item.get("categories", []),
                        "description":     item.get("description", ""),
                        "price_level":     item.get("priceLevel", ""),
                        "rating":          item.get("totalScore", ""),
                        "review_count":    item.get("reviewsCount", ""),
                        "address":         item.get("address", ""),
                        "amenities":       item.get("amenities", []),
                        "menu_highlights": menu_highlights,
                        "from_the_owner":  item.get("fromTheOwner", ""),
                        "photo_urls":      photo_urls,
                    }
                except Exception as bp_err:
                    brand_profile = {"name": name}

            # ── Reviews ──
            for review in item.get("reviews", []):
                if not isinstance(review, dict):
                    continue
                text = review.get("text") or ""
                text = text.strip()
                if text:
                    reviews.append(text)

        return reviews, brand_profile

    except Exception as exc:
        st.warning(f"Apify fetch failed: {exc}. Using demo data.")
        return [], {}


# Convenience wrapper for competitor fetch (reviews only)
def fetch_apify_reviews(name, location, max_reviews=50):
    reviews, _ = fetch_place_data(name, location, max_reviews)
    return reviews


def analyze_brand_visuals(photo_urls, restaurant_name):
    """
    Send up to 5 Google Maps photos to GPT-4o Vision.
    Returns a visual brand brief string describing the brand aesthetic.
    """
    if not photo_urls:
        return ""

    # Build vision message — system + images + question
    image_parts = []
    for url in photo_urls[:5]:
        try:
            # Verify URL is reachable before sending to vision
            r = requests.head(url, timeout=5)
            if r.status_code == 200:
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"},
                })
        except Exception:
            continue

    if not image_parts:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                *image_parts,
                {
                    "type": "text",
                    "text": (
                        f"These are real photos from {restaurant_name}'s Google Maps listing. "
                        "Analyze them as a brand consultant and describe: "
                        "(1) The signature drinks or dishes visible — be specific about what you see "
                        "(e.g. dark green ribbed ceramic mugs, iced drinks with cream on top, shawarma wraps in parchment). "
                        "(2) The interior vibe and aesthetic — surfaces, lighting, furniture style, color palette. "
                        "(3) The brand color accents visible — in cups, walls, signage, packaging. "
                        "(4) The overall mood — casual/upscale/cozy/modern/rustic/vibrant. "
                        "(5) What makes this place visually distinct from a generic version of the same type of restaurant. "
                        "Be specific and visual. Under 120 words. No bullet points — write as a flowing brand brief."
                    ),
                },
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return ""


DEMO_OWN_REVIEWS = [
    "Amazing food and lightning-fast service — was in and out in 20 minutes during lunch rush.",
    "The portions are huge! Definitely worth the price.",
    "Friendly staff who remembered my order from last time. Love this place.",
    "Best tacos in town, fresh ingredients every time.",
    "Service was incredibly quick even on a Friday night. Highly recommend.",
    "The food quality is consistently excellent — never had a bad meal here.",
    "Great atmosphere, cozy vibe, perfect for date night.",
    "A bit pricey but the quality makes it worth it.",
    "Parking can be tricky on weekends but the food makes up for it.",
    "Portions are generous and the flavor is always spot-on.",
]
DEMO_COMP_REVIEWS = [
    "Waited 45 minutes for our food on a Tuesday. Completely unacceptable.",
    "Staff was rude when we asked about a missing item. Won't be back.",
    "Food arrived cold both times I've visited. Disappointing.",
    "Overpriced for mediocre quality. Expected better.",
    "Service is painfully slow even when the restaurant isn't busy.",
    "Love the decor but the wait times kill the experience.",
    "Rude server made the whole night uncomfortable.",
    "Food was lukewarm and the bathroom was not clean.",
    "Always a long wait — they never seem to have enough staff.",
    "The food quality has really gone downhill. It used to be great.",
]

DEMO_BRAND_PROFILE = {
    "name":            "Demo Restaurant",
    "category":        "Restaurant",
    "categories":      ["Restaurant", "Cafe"],
    "description":     "A cozy neighborhood spot known for fresh ingredients and fast service.",
    "price_level":     "Moderate",
    "rating":          4.5,
    "review_count":    120,
    "address":         "123 Main St",
    "amenities":       ["Dine-in", "Takeout", "Delivery"],
    "menu_highlights": [],
    "from_the_owner":  "",
}


def build_marketing_prompt(restaurant_name, own_praise, own_complaints, comp_complaints, opportunities, num_posts=3, brand_profile=None):
    praise_str    = "\n".join(f"- {k} ({len(v)} mentions)" for k, v in own_praise.items())      or "None detected"
    complaint_str = "\n".join(f"- {k} ({len(v)} mentions)" for k, v in own_complaints.items())  or "None detected"
    comp_weak_str = "\n".join(f"- {k} ({len(v)} mentions)" for k, v in comp_complaints.items()) or "No competitor data"
    opp_str       = "\n".join(f"- {o}" for o in opportunities) or "None identified"
    bp            = brand_profile or {}

    # Build brand brief from profile data
    brand_lines = []
    if bp.get("category"):
        brand_lines.append(f"Category: {bp['category']}")
    if bp.get("categories"):
        brand_lines.append(f"All categories: {', '.join(bp['categories'])}")
    if bp.get("price_level"):
        brand_lines.append(f"Price level: {bp['price_level']}")
    if bp.get("rating"):
        brand_lines.append(f"Google rating: {bp['rating']} ({bp.get('review_count', '?')} reviews)")
    if bp.get("description"):
        brand_lines.append(f"Business description: {bp['description']}")
    if bp.get("from_the_owner"):
        brand_lines.append(f"From the owner: {bp['from_the_owner']}")
    if bp.get("menu_highlights"):
        brand_lines.append(f"Menu highlights: {', '.join(bp['menu_highlights'])}")
    if bp.get("amenities"):
        brand_lines.append(f"Amenities: {', '.join(bp['amenities'][:8])}")
    brand_str = "\n".join(brand_lines) or "No brand profile data available."

    # Visual brief from GPT-4o Vision analysis of actual brand photos
    visual_brief = bp.get("visual_brief", "")
    visual_section = f"\nVISUAL BRAND BRIEF (from analyzing actual photos of {restaurant_name}):\n{visual_brief}" if visual_brief else ""

    return inspect.cleandoc(f"""
    You are a restaurant marketing strategist and brand expert. Generate ready-to-use marketing content for "{restaurant_name}".

    BRAND PROFILE (study this carefully before creating any content):
    {brand_str}{visual_section}

    WHAT CUSTOMERS LOVE:
    {praise_str}

    WHAT CUSTOMERS COMPLAIN ABOUT (top issues only):
    {complaint_str}

    COMPETITOR WEAKNESSES:
    {comp_weak_str}

    COMPETITIVE OPPORTUNITIES:
    {opp_str}

    Return ONLY valid JSON, no markdown fences:
    {{
      "executive_summary": "2-3 sentence strategic overview",
      "positioning_statement": "One punchy sentence positioning {restaurant_name} vs competitors",
      "instagram_posts": [
        /* Generate exactly {num_posts} post object(s) in this array. Each object:
        {{
          "caption": "full ready-to-post caption with emojis and hashtags",
          "image_prompt": "Write a SHORT, specific flux-realism photo prompt under 25 words. Use the visual brand brief above to be specific about this exact restaurant. For shawarma: chicken shawarma wrap cross-section on parchment, garlic sauce, fresh herbs. For coffee: specific drink style from brand brief on exact surface. Always add: Fujifilm XT4, f/1.8, film grain, no text, no people.",
          "rationale": "why this works strategically for Instagram engagement"
        }}*/
      ],
      "google_business_posts": [
        {{"post": "...", "goal": "what this achieves"}},
        {{"post": "...", "goal": "..."}}
      ],
      "counter_move_content": [
        {{"hook": "competitor weakness", "content": "ready-to-post counter copy"}},
        {{"hook": "...", "content": "..."}}
      ],
      "promotional_campaigns": [
        {{"name": "...", "idea": "1-2 sentences", "channel": "Instagram / Google / Email / etc"}},
        {{"name": "...", "idea": "...", "channel": "..."}},
        {{"name": "...", "idea": "...", "channel": "..."}}
      ],
      "top_actions": ["Action 1", "Action 2", "Action 3"]
    }}
    """)


def generate_poster(image_prompt, tagline, restaurant_name):
    """Generate image via Pollinations flux-realism, overlay clean text."""
    import urllib.parse
    import re as _re

    try:
        # 1. Build a sharp, negative-prompted Pollinations URL
        # Core prompt — clean and specific
        # Replace forward slashes in prompt — they break URL path parsing
        clean_image_prompt = image_prompt.strip().replace("f/", "f").replace("/", " ")
        full_prompt = (
            f"{clean_image_prompt} "
            "Photorealistic food photography. Editorial Instagram aesthetic. "
            "No text, no words, no letters, no watermarks, no logos anywhere in the image."
        )

        # Negative prompt — aggressively strip stock photo feel
        negative = "text, watermark, logo, cartoon, blurry, deformed, ugly, oversaturated, people smiling at camera"

        encoded_prompt   = urllib.parse.quote(full_prompt)
        encoded_negative = urllib.parse.quote(negative)
        seed = abs(hash(restaurant_name + tagline)) % 99999

        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?model=flux-realism"
            f"&negative={encoded_negative}"
            f"&width=1080&height=1080"
            f"&nologo=true"
            f"&seed={seed}"
        )

        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")

        # 2. Gradient — bottom 40%, max alpha 170
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        w, h    = img.size
        gs      = int(h * 0.60)
        for y in range(gs, h):
            a = int(170 * ((y - gs) / (h - gs)))
            draw_ov.line([(0, y), (w, y)], fill=(0, 0, 0, a))
        img = Image.alpha_composite(img, overlay).convert("RGB")

        # 3. Fonts — macOS first, Linux fallback, then default
        def load_font(candidates, size):
            for path in candidates:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
            return ImageFont.load_default()

        font_name = load_font([
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ], 58)
        font_tag = load_font([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ], 28)

        draw = ImageDraw.Draw(img)

        # 4. Strip emoji — only ASCII burns cleanly onto poster
        def strip_emoji(text):
            return _re.sub(r'[^\x00-\x7F]+', '', text).strip()

        # Restaurant name — white, bold, bottom-left
        draw.text((54, h - 148), strip_emoji(restaurant_name.upper()), font=font_name, fill=(255, 255, 255))

        # Tagline — max 6 words, parchment tone
        short_tag = " ".join(strip_emoji(tagline).split()[:6])
        draw.text((56, h - 72), short_tag, font=font_tag, fill=(242, 234, 216))

        # 5. Return as PNG bytes
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as exc:
        st.warning(f"Poster generation failed: {exc}")
        return None


def sec(label, cls=""):
    st.markdown(
        f'<div class="sec-head"><span class="sec-head-label {cls}">{label}</span>'
        f'<div class="sec-head-rule"></div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# MAIN FLOW — (A) analysis on button click, (B) rendering persists across reruns
# ─────────────────────────────────────────────────────────────────────────────

# A. Run analysis only when button is clicked
if run_btn:
    if not restaurant_name.strip():
        st.warning("Enter a restaurant name to get started.")
        st.stop()

    name_display = restaurant_name.strip()

    # Always clear content state on every new analysis run
    for k in ["post_briefs", "posters_generated", "poster_cache", "marketing_data"]:
        st.session_state.pop(k, None)
    # Also clear widget key state so text_area/text_input reset
    for k in list(st.session_state.keys()):
        if k.startswith("img_prompt_") or k.startswith("headline_") or k.startswith("caption_"):
            del st.session_state[k]

    # 1. Collect reviews + brand profile + visual analysis
    with st.spinner("Studying the brand and pulling reviews..."):
        if APIFY_AVAILABLE:
            own_texts, brand_profile = fetch_place_data(restaurant_name, restaurant_location)
            comp_texts, _            = fetch_place_data(competitor_name, competitor_location) if competitor_name.strip() else ([], {})
            if not own_texts:
                st.info("No live reviews found -- showing demo output.")
                own_texts     = DEMO_OWN_REVIEWS
                comp_texts    = DEMO_COMP_REVIEWS
                brand_profile = DEMO_BRAND_PROFILE
        else:
            st.info("No APIFY_API_TOKEN found -- showing demo output.")
            own_texts     = DEMO_OWN_REVIEWS
            comp_texts    = DEMO_COMP_REVIEWS if competitor_name.strip() else []
            brand_profile = DEMO_BRAND_PROFILE

    # 2. Vision analysis — study actual brand photos
    photo_urls = brand_profile.get("photo_urls", [])
    visual_brief = ""
    if photo_urls:
        with st.spinner("Analyzing brand visuals..."):
            visual_brief = analyze_brand_visuals(photo_urls, restaurant_name)
            if visual_brief:
                brand_profile["visual_brief"] = visual_brief

    # 2. Sentiment + themes
    df_own  = build_sentiment_df(own_texts,  name_display)
    df_comp = build_sentiment_df(comp_texts, competitor_name or "Competitor")

    own_praise      = extract_themes(df_own,  "Positive")
    own_complaints  = extract_themes(df_own,  "Negative")
    comp_complaints = extract_themes(df_comp, "Negative")
    opportunities   = [
        f"{t}: competitors struggle here, you are praised for it"
        for t in set(own_praise) & set(comp_complaints)
    ]

    # 3. LLM marketing content
    prompt = build_marketing_prompt(name_display, own_praise, own_complaints, comp_complaints, opportunities, num_posts, brand_profile)
    with st.spinner("Generating marketing content..."):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw  = resp.choices[0].message.content or ""
            raw  = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
            raw  = re.sub(r"\n?```$",        "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(raw)
        except json.JSONDecodeError:
            st.error("Unexpected response format.")
            st.code(raw)
            st.stop()
        except Exception as exc:
            st.error(f"Content generation failed: {exc}")
            st.stop()

    # 4. Persist everything to session_state
    vc = df_own["sentiment_label"].value_counts() if not df_own.empty else pd.Series(dtype=int)
    st.session_state.update({
        "analyzed_for":    name_display,
        "brand_profile":   brand_profile,
        "df_own":          df_own,
        "df_comp":         df_comp,
        "marketing_data":  data,
        "own_praise":      own_praise,
        "own_complaints":  own_complaints,
        "comp_complaints": comp_complaints,
        "opportunities":   opportunities,
        "stat_pos":        int(vc.get("Positive", 0)),
        "stat_neg":        int(vc.get("Negative", 0)),
        "stat_neu":        int(vc.get("Neutral",  0)),
        "total_reviews":   len(df_own),
        "opp_count":       len(opportunities),
    })
    ig_posts = data.get("instagram_posts", [])
    st.session_state["post_briefs"] = [
        {
            "caption":      p.get("caption", ""),
            "image_prompt": p.get("image_prompt", ""),
            "headline":     p.get("caption", "").split(".")[0].split("!")[0].split("\n")[0][:60],
            "rationale":    p.get("rationale", ""),
        }
        for p in ig_posts
    ]
    st.session_state.pop("posters_generated", None)
    st.session_state.pop("poster_cache", None)


# B. Render results -- always visible once data exists, survives reruns
# Show Gemini debug info if it failed to init
if not GEMINI_AVAILABLE:
    err = globals().get("_GEMINI_ERR", "Unknown error")
    st.sidebar.error(f"Gemini unavailable: {err}")
    st.sidebar.code(f"GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY'))}")

if "marketing_data" not in st.session_state:
    st.stop()

data            = st.session_state["marketing_data"]
name_display    = st.session_state["analyzed_for"]
own_praise      = st.session_state["own_praise"]
own_complaints  = st.session_state["own_complaints"]
comp_complaints = st.session_state["comp_complaints"]
opportunities   = st.session_state["opportunities"]
df_own          = st.session_state["df_own"]

# Stats strip
st.markdown(f"""
<div class="stat-strip">
  <div class="stat-box tot"><div class="sl">Reviews</div><div class="sv">{st.session_state['total_reviews']}</div></div>
  <div class="stat-box pos"><div class="sl">Positive</div><div class="sv">{st.session_state['stat_pos']}</div></div>
  <div class="stat-box neg"><div class="sl">Negative</div><div class="sv">{st.session_state['stat_neg']}</div></div>
  <div class="stat-box neu"><div class="sl">Neutral</div><div class="sv">{st.session_state['stat_neu']}</div></div>
  <div class="stat-box opp"><div class="sl">Opportunities</div><div class="sv">{st.session_state['opp_count']}</div></div>
</div>
""", unsafe_allow_html=True)

# Brand profile card
bp = st.session_state.get("brand_profile", {})
if bp:
    sec("Brand studied")

    # Visual brief — hero card if available
    if bp.get("visual_brief"):
        st.markdown(f"""
        <div class="ccard" style="border-color:var(--amber);margin-bottom:1rem">
          <div class="cl">Visual brand brief — analyzed from actual photos</div>
          <div class="cb" style="font-size:0.88rem;line-height:1.7">{bp["visual_brief"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actual brand photos pulled from Google Maps
    photo_urls = bp.get("photo_urls", [])
    if photo_urls:
        photo_cols = st.columns(min(len(photo_urls), 5), gap="small")
        for i, url in enumerate(photo_urls[:5]):
            with photo_cols[i]:
                try:
                    st.image(url, use_container_width=True)
                except Exception:
                    pass
        st.markdown("<br>", unsafe_allow_html=True)

    # Meta info row
    bp_col1, bp_col2, bp_col3 = st.columns(3, gap="medium")
    with bp_col1:
        if bp.get("category"):
            pl = f"<div class='cr'>{bp.get('price_level','')}</div>" if bp.get("price_level") else ""
            st.markdown(f"""
            <div class="ccard">
              <div class="cl">Category</div>
              <div class="cb">{bp.get("category","")}</div>
              {pl}
            </div>
            """, unsafe_allow_html=True)
    with bp_col2:
        desc = bp.get("description") or bp.get("from_the_owner","")
        if desc:
            st.markdown(f"""
            <div class="ccard">
              <div class="cl">About</div>
              <div class="cb" style="font-size:0.85rem">{desc[:200]}{"..." if len(desc)>200 else ""}</div>
            </div>
            """, unsafe_allow_html=True)
    with bp_col3:
        highlights = bp.get("menu_highlights", [])
        amenities  = bp.get("amenities", [])[:5]
        items      = highlights[:5] if highlights else amenities
        if items:
            label = "Menu highlights" if highlights else "Amenities"
            st.markdown(f"""
            <div class="ccard">
              <div class="cl">{label}</div>
              <div class="cb" style="font-size:0.85rem">{"<br>".join(items)}</div>
            </div>
            """, unsafe_allow_html=True)

# Intelligence columns
ic1, ic2 = st.columns(2, gap="large")
with ic1:
    sec("What customers love", "pos")
    if own_praise:
        pills = "".join(f'<span class="pill pos">{t}</span>' for t in own_praise)
        st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
        for _, examples in list(own_praise.items())[:2]:
            st.markdown(f'<div class="qline pos">{examples[0]}</div>', unsafe_allow_html=True)
    else:
        st.caption("No positive themes detected.")

with ic2:
    sec("What needs work")
    if own_complaints:
        top3 = list(own_complaints.items())[:3]
        pills = "".join(f'<span class="pill neg">{t}</span>' for t, _ in top3)
        st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
        for _, examples in top3:
            st.markdown(f'<div class="qline neg">{examples[0][:120]}...</div>', unsafe_allow_html=True)
    else:
        st.caption("No complaint themes detected.")

if comp_complaints:
    sec("Competitor weak spots", "comp")
    pills = "".join(f'<span class="pill opp">{t}</span>' for t in comp_complaints)
    st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)
if opportunities:
    for opp in opportunities:
        st.markdown(f'<div class="qline opp">{opp}</div>', unsafe_allow_html=True)

# Executive summary
exec_sum = data.get("executive_summary", "")
position = data.get("positioning_statement", "")
if exec_sum:
    sec("Intelligence summary")
    pos_html = f"<div class='cr'>Positioning: {position}</div>" if position else ""
    st.markdown(f"""
    <div class="ccard" style="border-color:var(--amber)">
      <div class="cl">Executive Summary</div>
      <div class="cb">{exec_sum}</div>
      {pos_html}
    </div>
    """, unsafe_allow_html=True)

# Instagram brief -> edit -> generate posters
briefs = st.session_state.get("post_briefs", [])
if briefs:
    sec("Instagram posts")
    st.markdown("""
    <div class="ccard" style="border-color:var(--amber);margin-bottom:1.5rem">
      <div class="cl">Review and edit before generating</div>
      <div class="cb" style="font-size:0.85rem;color:var(--text-soft)">
        Each post shows what will be created. Edit anything below, then click
        <strong style="color:var(--amber)">Generate all posters</strong> when ready.
      </div>
    </div>
    """, unsafe_allow_html=True)

    for i, brief in enumerate(briefs):
        st.markdown(f"""
        <div class="sec-head" style="margin-top:1.5rem">
          <span class="sec-head-label">Post {i+1}</span>
          <div class="sec-head-rule"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ccard" style="margin-bottom:0.75rem;background:var(--raised)">
          <div class="cl">What will be generated</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:0.25rem">
            <div>
              <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--slate);margin-bottom:0.2rem">Image style</div>
              <div style="font-size:0.83rem;color:var(--text-soft)">{brief["image_prompt"][:110]}...</div>
            </div>
            <div>
              <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--slate);margin-bottom:0.2rem">Text on poster</div>
              <div style="font-size:0.83rem;color:var(--text)"><strong>{name_display.upper()}</strong><br><span style="color:var(--text-soft)">{brief["headline"]}</span></div>
            </div>
          </div>
          <div style="margin-top:0.65rem">
            <div style="font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--slate);margin-bottom:0.2rem">Strategy</div>
            <div style="font-size:0.8rem;color:var(--text-soft);font-style:italic">{brief["rationale"]}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ec1, ec2 = st.columns(2, gap="medium")
        with ec1:
            briefs[i]["image_prompt"] = st.text_area(
                f"Image style -- post {i+1}",
                value=brief["image_prompt"],
                height=100,
                key=f"img_prompt_{name_display}_{i}",
                help="Describe the DALL-E photo. Be specific about mood, lighting, subject.",
            )
            briefs[i]["headline"] = st.text_input(
                f"Tagline on poster -- post {i+1}",
                value=brief["headline"],
                key=f"headline_{name_display}_{i}",
                help="Short text overlaid on the bottom of the image.",
            )
        with ec2:
            briefs[i]["caption"] = st.text_area(
                f"Caption -- post {i+1}",
                value=brief["caption"],
                height=140,
                key=f"caption_{name_display}_{i}",
                help="Full Instagram caption with emojis and hashtags.",
            )
        st.markdown("<br>", unsafe_allow_html=True)

    st.session_state["post_briefs"] = briefs

    if st.button("Generate all posters ->", key="gen_posters"):
        st.session_state["posters_generated"] = True
        st.session_state.pop("poster_cache", None)

    if st.session_state.get("posters_generated"):
        sec("Generated posters")
        if "poster_cache" not in st.session_state:
            st.session_state["poster_cache"] = {}

        for i, brief in enumerate(st.session_state["post_briefs"]):
            cache_key = f"{i}|{brief['image_prompt'][:40]}|{brief['headline']}"
            if cache_key not in st.session_state["poster_cache"]:
                with st.spinner(f"Creating poster {i+1} of {len(st.session_state['post_briefs'])}..."):
                    poster_bytes = generate_poster(brief["image_prompt"], brief["headline"], name_display)
                if poster_bytes:
                    st.session_state["poster_cache"][cache_key] = poster_bytes

            poster_bytes = st.session_state["poster_cache"].get(cache_key)
            pc1, pc2 = st.columns([1, 1.4], gap="large")
            with pc1:
                if poster_bytes:
                    st.image(poster_bytes, use_container_width=True)
                    st.download_button(
                        label=f"Download poster {i+1}",
                        data=poster_bytes,
                        file_name=f"dishradar_post_{i+1}.png",
                        mime="image/png",
                        key=f"dl_poster_{i}",
                    )
                else:
                    st.warning(f"Poster {i+1} failed to generate.")
            with pc2:
                st.markdown(f"""
                <div class="ccard" style="height:100%">
                  <div class="cl">Caption -- post {i+1}</div>
                  <div class="cb">{brief["caption"]}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# Google Business posts
gb_posts = data.get("google_business_posts", [])
if gb_posts:
    sec("Google Business posts")
    for post in gb_posts:
        st.markdown(f"""
        <div class="ccard">
          <div class="cl">Google Business Update</div>
          <div class="cb">{post.get('post','')}</div>
          <div class="cr">Goal: {post.get('goal','')}</div>
        </div>
        """, unsafe_allow_html=True)

# Counter-move content
counter = data.get("counter_move_content", [])
if counter:
    sec("Competitor counter-moves", "comp")
    for item in counter:
        st.markdown(f"""
        <div class="ccard comp-c">
          <div class="cl">Countering: {item.get('hook','')}</div>
          <div class="cb">{item.get('content','')}</div>
        </div>
        """, unsafe_allow_html=True)

# Promotional campaigns
promos = data.get("promotional_campaigns", [])
if promos:
    sec("Campaign ideas")
    pcols = st.columns(min(len(promos), 3), gap="medium")
    for i, promo in enumerate(promos):
        with pcols[i % 3]:
            st.markdown(f"""
            <div class="ccard promo-c">
              <div class="cl">{promo.get('channel','')}</div>
              <div class="cb" style="font-weight:600;margin-bottom:0.35rem">{promo.get('name','')}</div>
              <div class="cr" style="font-style:normal">{promo.get('idea','')}</div>
            </div>
            """, unsafe_allow_html=True)

# Top actions
actions = data.get("top_actions", [])
if actions:
    sec("Prioritised actions")
    for i, action in enumerate(actions, 1):
        st.markdown(f"""
        <div class="action-row">
          <div class="action-num">{i}</div>
          <div class="action-txt">{action}</div>
        </div>
        """, unsafe_allow_html=True)

# Raw data
st.markdown("<br>", unsafe_allow_html=True)
if not df_own.empty:
    with st.expander(f"Raw review data -- {name_display} ({len(df_own)} reviews)"):
        st.dataframe(df_own, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df_own.to_csv(index=False).encode("utf-8"),
        file_name="dishradar_reviews.csv",
        mime="text/csv",
    )
