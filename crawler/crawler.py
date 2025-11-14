import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def fetch_page_text(url: str) -> str | None:
    """
    Fetches the URL and extracts ALL clean text from the <body>
    [or all whole page if there is no <body>].
    """
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            if soup.body:
                page_text = soup.body.get_text(separator=" ", strip=True)
            else:
                page_text = soup.get_text(separator=" ", strip=True)

            cleaned_text = " ".join(page_text.split())
            return cleaned_text

        else:
            print(f"Error (Status {response.status_code}) while accessing {url}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None


def main():
    input_file = '../URL_list.csv'
    output_file = 'crawled_data/200_pages_data.json'
    max_pages = 200

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    if 'max(page)' not in df.columns:
        print(f"Error: Column 'max(page)' not found in {input_file}. Found columns: {df.columns.tolist()}")
        return

    # Get unique, non-null URLs, limited by max_pages
    urls_to_crawl = df['max(page)'].dropna().unique()[:max_pages]

    print(f"Starting to scrape text from {len(urls_to_crawl)} URLs...")

    crawled_data = []
    for i, url in enumerate(urls_to_crawl):
        print(f"[{i + 1}/{len(urls_to_crawl)}] Processing: {url}")
        text = fetch_page_text(url)

        if text:
            crawled_data.append({'url': url, 'text': text})

        time.sleep(1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=4)

    print(f"\nScraping finished. Data saved to {output_file}")
    print(f"Collected {len(crawled_data)} pages.")


if __name__ == "__main__":
    main()