# 🍽️ DishRadar AI
### AI Marketing Intelligence Platform for Restaurants

> Turn customer reviews and competitor weaknesses into ready-to-post marketing content.

DishRadar automatically pulls real Google Maps reviews, studies the brand's visual identity, analyzes sentiment and competitive gaps, and generates Instagram posts, Google Business updates, and downloadable 1080x1080 posters — all from a restaurant name and city.

---

## What it does

1. **Studies the brand** — pulls the Google Maps listing, extracts business description, category, price level, amenities, and up to 6 photos. Sends the photos to GPT-4o Vision to build a visual brand brief.
2. **Collects reviews** — scrapes up to 50 real Google Maps reviews via Apify.
3. **Analyzes sentiment** — classifies every review as Positive, Neutral, or Negative using VADER.
4. **Extracts themes** — maps reviews across 8 categories: Food Quality, Service, Wait Time, Price, Portion Size, Atmosphere, Cleanliness, Parking.
5. **Finds competitive gaps** — identifies where a competitor's weaknesses align with the restaurant's strengths.
6. **Generates marketing content** — Instagram captions, Google Business posts, counter-move copy, promotional campaign ideas, and prioritised actions.
7. **Creates posters** — generates a 1080x1080 image via Pollinations flux-realism, overlays restaurant name and tagline, makes it downloadable.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Review scraping | Apify — `compass/crawler-google-places` |
| Sentiment analysis | VADER |
| Brand vision analysis | OpenAI GPT-4o-mini (vision) |
| Content generation | OpenAI GPT-4.1-mini |
| Image generation | Pollinations AI — flux-realism model (free, no key) |
| Image compositing | Pillow (PIL) |
| Data processing | Pandas |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourname/dishradar_ai.git
cd dishradar_ai
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

> Python 3.11 recommended. Python 3.9 works but requires `--prefer-binary` flag when installing some packages.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
APIFY_API_TOKEN=apify_api_your-token-here
```

Get your keys:
- **OpenAI** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Apify** — [console.apify.com](https://console.apify.com) → Settings → Integrations → API token

> Pollinations AI requires no API key. Image generation works out of the box.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Enter your **restaurant name** and **city** in the left column
2. Optionally enter a **competitor name** and city in the right column
3. Choose how many **Instagram posts** to generate (1, 2, or 3)
4. Click **Generate content →**
5. Review the brand profile, sentiment summary, and intelligence sections
6. Review and edit each post brief — image style, tagline, caption
7. Click **Generate all posters →**
8. Download your 1080x1080 PNG posters

---

## Project structure

```
dishradar_ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (never commit this)
├── .gitignore              # Excludes .env and venv
└── README.md
```

---

## .gitignore

Make sure your `.gitignore` includes:

```
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

## Environment variables reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Powers content generation and brand vision analysis |
| `APIFY_API_TOKEN` | Yes | Powers Google Maps review scraping |
| `GEMINI_API_KEY` | Optional | For Imagen 3 image generation via Google AI Studio (requires billing) |

---

## How the content generation works

DishRadar sends a single structured prompt to GPT-4.1-mini that includes:

- The brand profile (category, description, price level, visual brief)
- Top praised themes from sentiment analysis
- Top complaint themes
- Competitor weakness themes
- Competitive opportunities (intersection of your strengths and their weaknesses)

The LLM returns a JSON object with all marketing assets. The image prompt inside that JSON is specifically written for flux-realism — short, precise, cuisine-specific, with exact surface, light source, and camera language.

---

## Known limitations

- **Image quality** — Pollinations flux-realism is good but not Instagram-native quality. Enabling billing on Google AI Studio unlocks Imagen 3 which produces significantly better food photography.
- **Review volume** — Apify free tier returns a limited number of reviews per run. Meaningful analysis benefits from 20+ reviews.
- **Theme extraction** — currently regex-based. Misses nuanced language. A future version will use LLM-based theme extraction.
- **No persistence** — all data lives in session state. Refreshing the page resets everything.

---

## Roadmap

- [ ] Host on Streamlit Cloud for shareable URL access
- [ ] Switch theme extraction to LLM-based for accuracy
- [ ] Add Facebook post generation
- [ ] Add persistent storage (Supabase/Postgres) for saved reports
- [ ] Multi-restaurant agency dashboard
- [ ] Scheduled content generation

---

## License

MIT
