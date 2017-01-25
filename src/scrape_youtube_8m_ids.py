# script to scrape YouTube IDs from YouTube-8M dataset (we want to download the videos manually for this project)
# adapted from http://codegists.com/snippet/python/youtube-8m-video-id-scraperpy_josephredfern_python

import os
import requests
from collections import defaultdict

os.chdir("../data/youtube-8m-id-scrape")

csv_prefix = "http://www.yt8m.org/csv"

r = requests.get("{0}/verticals.json".format(csv_prefix))
verticals = r.json()

block_urls = defaultdict(list)
count = 0
for cat, urls in verticals.items():
    for url in urls:
        jsurl = "{0}/j/{1}.js".format(csv_prefix, url.split("/")[-1])
        block_urls[cat[1:]].append(jsurl)
        count += 1  # lazy.

ids_by_cat = defaultdict(list)

downloaded = 0.0
for cat_name, block_file_urls in block_urls.items():
    for block_file_url in block_file_urls:
        print("[{0}%] Downloading block file: {1} {2}".format((100.0 * downloaded / count), block_file_url, cat_name))
        try:
            r = requests.get(block_file_url)
            idlist = r.content.split("\"")[3]
            ids = [n for n in idlist.split(";") if len(n) > 3]
            ids_by_cat[cat_name] += ids
        except IndexError, IOError:
            print("Failed to download or process block at {0}".format(block_file_url))
        downloaded += 1  # increment even if we've failed.

    with open("{0}.txt".format(cat_name), "w") as idfile:
        print("Writing ids to {0}.txt".format(cat_name))
        for vid in ids_by_cat[cat_name]:
            idfile.write("{0}\n".format(vid))
        print("Done.")