import requests
# import urllib.request
import time
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import os


def extract(link, image_name, image_path):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        #         print(soup.prettify())
        image_link = soup.find(property="og:image")['content']
        image = requests.get(image_link, stream=True).content

        with open(f'{image_path}/{image_name}', 'wb') as handler:
            handler.write(image)
    except Exception as e:
        print("IMAGE ERROR ", e)
        image_name = np.nan

    try:
        dutch = soup.find(text='Title of the art-work in Dutch')
        dutch = dutch.find_parent('dt').find_next_sibling('dd')
        dutch = dutch.text.strip()
    except Exception as e:
        print("DUTCH ERROR ", e)
        dutch = np.nan

    try:
        english = soup.find(text='English title')
        english = english.find_parent('dt').find_next_sibling('dd')
        english = english.text.strip()
    except Exception as e:
        print("ENGLISH ERROR ", e)
        english = np.nan

    try:
        iconclass = ''
        for item in soup.find_all('span', 'text'):
            if "Iconclass" in item.text:
                if iconclass != '':
                    iconclass += ';'
                iconclass += item.find('em').text
        iconclass = np.nan if iconclass == '' else iconclass
    except Exception as e:
        print("ICONCLASS ERROR ", e)
        iconclass = np.nan

    image_name = image_name if os.path.isfile(f'{image_path}/{image_name}') else np.nan

    return link, dutch, english, iconclass, image_name


if __name__ == '__main__':
    for i in range(232258, 255200):
        record = extract(f"https://rkd.nl/en/explore/images/record?query=&start={i}", f'{i}.jpg', 'rkd_images')
        frame = pd.DataFrame([record], columns=['link', 'dutch', 'english', 'iconclass', 'image'])
        if i == 0:
            frame.to_csv('rkd_images_results.csv', sep='\t')
        else:
            with open('rkd_images_results.csv', 'a') as f:
                frame.to_csv(f, header=False, sep='\t')

        if i%1000 == 0:
            print(i)

        time.sleep(0.4)
