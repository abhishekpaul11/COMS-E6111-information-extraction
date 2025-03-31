import requests


def google_search(query, api_key, search_engine_id):
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        'q': query,
        'key': api_key,
        'cx': search_engine_id,
        'num': 10
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None