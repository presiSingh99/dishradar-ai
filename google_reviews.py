import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def find_place(query):
    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
    )

    params = {
        "query": query,
        "key": API_KEY
    }

    response = requests.get(url, params=params)

    data = response.json()

    if not data["results"]:
        return None

    return data["results"][0]["place_id"]

def get_reviews(place_id):

    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
    )

    params = {
        "place_id": place_id,
        "fields": "reviews,name",
        "key": API_KEY
    }

    response = requests.get(url, params=params)

    data = response.json()

    reviews = []

    for review in data["result"].get("reviews", []):

        reviews.append(
            review["text"]
        )

    return reviews

if __name__ == "__main__":

    place_id = find_place("Chipotle Silver Spring MD")

    print("Place ID:", place_id)

    if place_id:
        reviews = get_reviews(place_id)

        print("\nReviews:\n")
        for review in reviews:
            print("-", review)
