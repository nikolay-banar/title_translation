import requests
# import urllib.request
import time
import pandas as pd


def extract_title(link):
    response = requests.get(link).text
    start = response.find("objectTitle")
    end = response.find("\n", start)
    title = response[start:end]
    quotes = [pos for pos, char in enumerate(title) if char == '"']
    return title[quotes[0] + 1: quotes[-1]]

if __name__ == '__main__':
    data = pd.read_csv('dutch_metadata.csv')
    for row in data.iterrows():
        old_url = row[1]['link']
        url = old_url.replace('rijksmuseum.nl/nl/collectie', 'rijksmuseum.nl/en/collection')
        try:
            object_title = extract_title(url)
        except Exception as e:
            print(url, '=====', e)

        with open("results.txt", "a") as my_file:
            str_to_append = old_url + '\t' + object_title + '\n'
            my_file.write(str_to_append)
            time.sleep(0.4)