import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os

def get_icd9_description(code):
    """
    Fetches the ICD-9 description for a given code from http://icd9.chrisendres.com/
    SEARCH URL: http://icd9.chrisendres.com/index.php?action=search&srchtext={code}
    """
    url = f"http://icd9.chrisendres.com/index.php?action=search&srchtext={code}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Pattern 1: Look for div with class 'dlvl'
            results = soup.find_all('div', class_='dlvl')
            for res in results:
                text = res.get_text(strip=True)
                # The text usually starts with the code
                if text.startswith(str(code)):
                    # Remove the code from the start.
                    # Example: "428 Heart failure" -> "Heart failure"
                    return text[len(str(code)):].strip()

            # Pattern 2: Sometimes it might be in a different structure if it's a sub-code
            # Try to find the exact code followed by text
            # This is a fallback
            
            return None
        else:
            print(f"Failed to fetch {code}: Status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching {code}: {e}")
        return None

def scrape_top_codes(codes):
    descriptions = {}
    for code in codes:
        print(f"Scraping code: {code}...")
        desc = get_icd9_description(code)
        if desc:
            print(f"Found: {desc}")
            descriptions[code] = desc
        else:
            print(f"Not Found for {code}")
            descriptions[code] = "Description not found"
        time.sleep(1) # Ethical delay
    return descriptions