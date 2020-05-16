from pathlib import Path
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

image_root_dir = Path('image')


max_num = 1000
for keyword in [
        "dog",
        "cat",
        "man",
        "woman",
        "car",
        "bicycle",
    ]:
    filters = {
        'type': 'photo'
    }
    crawler = GoogleImageCrawler(
        storage={'root_dir': image_root_dir / keyword},
        # parser_threads=2,
        # downloader_threads=4,
        )
    crawler.crawl(
        keyword=keyword,
        max_num=max_num,
        # filters=filters
        )
