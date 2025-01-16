import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random

# Filmin adı
movie_name = "Mad_Max_Fury_Road_500_Reviews"

# WebDriver ayarları (headless mode)
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920x1080')
options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)
driver.get('https://www.rottentomatoes.com/m/mad_max_fury_road/reviews?type=user')

review_count = 0
max_reviews = 500  # Çekmek istediğin maksimum yorum sayısı

# CSV dosyasını oluştur
with open(f'{movie_name}_reviews.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # CSV dosyasına başlıkları yaz
    writer.writerow(['Name', 'Rating', 'Review'])

    # Load More butonuna tıklama döngüsü
    while review_count < max_reviews:
        try:
            load_more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'load-more-container'))
            )
            load_more_button.click()
            time.sleep(random.uniform(1, 3))  # Rastgele bekleme süresi
        except:
            break  # Load More butonu artık bulunamıyorsa çık

        # Sayfanın tamamını BeautifulSoup ile parse et
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Yorumları çek
        review_blocks = soup.find_all('div', class_='audience-review-row')
        for block in review_blocks[review_count:]:
            name_tag = block.find('span', class_='audience-reviews__name')
            review_tag = block.find('p', class_='audience-reviews__review')
            rating_tag = block.find('rating-stars-group')

            name = name_tag.get_text(strip=True) if name_tag else 'No Name'
            review_text = review_tag.get_text(strip=True) if review_tag else 'No Review'
            rating = rating_tag['score'] if rating_tag else 'No Rating'

            review_count += 1
            # Her yorumu CSV'ye kaydet
            writer.writerow([name, rating, review_text])

            print(f'{review_count}. Name: {name}\nRating: {rating}\nReview: {review_text}\n')

            if review_count >= max_reviews:
                break

# WebDriver'ı kapat
driver.quit()
