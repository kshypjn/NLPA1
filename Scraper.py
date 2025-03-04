from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from webdriver_manager.chrome import ChromeDriverManager
import re

# Saves the scraped articles to CTScraped.csv

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--headless')


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\x00-\x7F]+', '', text)  
    sentences = text.split('. ')
    cleaned_sentences = [sentence for sentence in sentences if 'also read' not in sentence.lower() and 'related' not in sentence.lower()]
    return '. '.join(cleaned_sentences).strip()

def is_incomplete_content(content):
    if len(content.split('. ')) < 3:  
        return True
    incomplete_phrases = [":", "Find below", "See the full list", "Here are"]
    last_sentence = content.strip().split('.')[-1]
    if any(phrase.lower() in last_sentence.lower() for phrase in incomplete_phrases):
        return True
    return False

def extract_sportstar_articles(existing_links, collected_articles):
    print("Extracting articles from Sportstar...")
    driver.get("https://sportstar.thehindu.com/cricket/champions-trophy/")
    time.sleep(5)

    while True:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        article_containers = soup.find_all('h3', class_='title')

        for article in article_containers:
            link_tag = article.find('a')
            if link_tag and link_tag.get('href'):
                title = link_tag.text.strip()
                link = 'https://sportstar.thehindu.com' + link_tag['href'] if not link_tag['href'].startswith('http') else link_tag['href']
                
                if link not in existing_links:
                    collected_articles.append({'title': title, 'url': link})
                    existing_links.add(link)
                    print(f"Collected article: {title}")
        
        
        try:
            show_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.small-link.see-more.center-align-load-more.down-view'))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_button)
            time.sleep(1)
            show_more_button.click()
            time.sleep(3)
        except:
            print("No more articles to load from Sportstar.")
            break
    
    return collected_articles

def extract_sportskeeda_articles(existing_links, collected_articles):
    print("Extracting articles from Sportskeeda...")
    base_url = "https://www.sportskeeda.com/go/icc-champions-trophy/news"
    page_number = 1
    max_pages = 2

    while page_number <= max_pages:
        url = f"{base_url}?page={page_number}"
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        article_containers = soup.select("a.feed-item-cta")

        for article in article_containers:
            title = article.text.strip()
            link = "https://www.sportskeeda.com" + article['href'] if not article['href'].startswith('http') else article['href']
            
            if link not in existing_links:
                collected_articles.append({'title': title, 'url': link})
                existing_links.add(link)
                print(f"Collected article: {title}")


        page_number += 1
        time.sleep(5)

    return collected_articles

def extract_content(url):
    print(f"Extracting content from {url}...")
    driver.get(url)
    time.sleep(random.uniform(3, 5))
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    content_div = None
    if "sportstar.thehindu.com" in url:
        content_div = soup.find('div', class_='articlebodycontent')
        if content_div:
            for tag in content_div.find_all(['div', 'ul'], class_=['comments-shares', 'related-topics-list']):
                tag.decompose()
            for tag in content_div.find_all(['p'], class_=['comments', 'related-topics-list', 'caption']):
                tag.decompose()
            for tag in content_div.find_all(['h2'], class_=['title-patch']):
                tag.decompose()
            for a_tag in content_div.select('p.related-topics-list a'):
                a_tag.decompose()
            for tag in content_div.find_all(['div', 'p', 'span'], class_=['inline_embed article-block-item', 'caption']):
                tag.decompose()
            for tag in content_div.find_all('p'):
                bold_tag = tag.find('b')
                link_tag = tag.find('a')
                if bold_tag and bold_tag.text.strip().lower() in ['also read', 'related']:
                    if link_tag and link_tag.get('href') and 'sportstar.thehindu.com' in link_tag['href']:
                        tag.decompose()
    elif "sportskeeda.com" in url:
        content_div = soup.find('div', id='article-content', class_='keeda_widget article-content-holder')
        if content_div:
            for tag in content_div.find_all(['div', 'span'], class_=['bottom-tagline bottom', 'post-rating-widget-parent', 'scrollable-content-holder article-padding', 'publisher-name']):
                tag.decompose()
            for img_tag in content_div.find_all('img'):
                img_tag.decompose()
            for tag in content_div.find_all('div', class_=['article-post-rating-widget', 'bottom-tagline', 'scrollable-content-holder', 'article-padding']):
                    tag.decompose()
            for tag in content_div.find_all('p'):
                if "Do you agree with" in tag.text or tag.get('class') == ['article-p']:
                    tag.decompose()
            for tag in content_div.find_all('div', class_=['mobile-p']):
                tag.decompose()
    
    if content_div:
        paragraphs = content_div.find_all('p')
        blockquotes = content_div.find_all('blockquote')
        content = ' '.join([clean_text(p.text) for p in paragraphs])
        blockquote_text = ' '.join([clean_text(bq.text) for bq in blockquotes])
        full_content = content + ' ' + blockquote_text if blockquote_text else content
        if is_incomplete_content(full_content):
            print("Discarding incomplete content.")
            return "Content extraction failed"
        if full_content.strip():
            print("Content successfully extracted.")
            return full_content
    
    print("Content extraction failed.")
    return "Content extraction failed"


print("Starting scraping process...")
existing_links = set()
collected_articles = []

collected_articles = extract_sportstar_articles(existing_links, collected_articles)
collected_articles = extract_sportskeeda_articles(existing_links, collected_articles)

valid_articles = []
for article in collected_articles:
    content = extract_content(article['url'])
    if content != "Content extraction failed":
        article['content'] = content
        valid_articles.append(article)
    time.sleep(random.uniform(1, 2))

dataframe = pd.DataFrame(valid_articles)
dataframe.to_csv("CTScraped.csv", index=False)
driver.quit()
print(f"Total articles added to CSV: {len(valid_articles)}")



