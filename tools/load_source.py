from pathlib import Path

import requests
from bs4 import BeautifulSoup
from html2text import HTML2Text

TARGET_WIKI_URL = "https://ja.wikipedia.org/wiki/メイドインアビス"


def load_source() -> None:
    response = requests.get(TARGET_WIKI_URL)

    soup = BeautifulSoup(response.text, "html.parser")
    html_content = str(soup.find(class_="mw-body-content"))

    text_maker = HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_images = True
    source = text_maker.handle(html_content)

    Path("assets/source.txt").write_text(source)

    return None


if __name__ == "__main__":
    load_source()
