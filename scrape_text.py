import requests
from bs4 import BeautifulSoup


def extract_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        raw_text = soup.get_text()

        print('\tCleaning the text (removing unnecessary white space and non-printable characters)')
        # Manual cleaning: Remove redundant newlines, tabs, and multiple spaces
        cleaned_text = " ".join(raw_text.split())  # Collapses all whitespace into single spaces
        # Remove common spurious characters (e.g., non-printable)
        cleaned_text = "".join(c for c in cleaned_text if c.isprintable() or c.isspace())

        if len(cleaned_text) == 0:
            return None

        if len(cleaned_text) > 10000:
            print(f'\tTrimming webpage content from {len(cleaned_text)} to 10000 characters')
        cleaned_text = cleaned_text[:10000]

        print(f'\tWebpage length (num characters): {len(cleaned_text)}')
        return cleaned_text
    except:
        return None