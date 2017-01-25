# script to download youtube videos from a list of IDs

from __future__ import unicode_literals
import os
import youtube_dl
# (https://github.com/rg3/youtube-dl/)

vid_list = [
    'GCx5oPWRKbc',  # no 1280x720 res, should error
    'sC9abcLLQpI',  # no 60fps, should error
    'rql_F8H3h9E',
]

# more info about options here:
# https://github.com/rg3/youtube-dl/blob/master/youtube_dl/YoutubeDL.py#L128-L278
ydl_opts = {
    # 'listformats': True,
    'ignoreerrors': True,
    # TODO use only 720p60fps video or not?
    'format': 'bestvideo[ext=mp4][width = 1280][height = 720][fps = 60]',
    # For debugging...
    # 'skip_download': True
}

# save all downloaded videos to proper folder
os.chdir("../data/youtube-8m-videos")

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(vid_list)
