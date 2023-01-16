import os
import sqlite3
import numpy as np
import pandas as pd
from multiprocessing import Pool
from urllib.request import Request, urlopen
from PIL import Image
DATA_DIR = "data/images"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
N_WORKERS = 64


### Remove data from disk

# class Downloader():
#   def download(self, df):
#     for i, row in df.iterrows():
#         id, url = row['id'], row['image_ship_url']
#         filepath = os.path.join(DATA_DIR, str(id)+'.jpg')
#         try:
#             with Image.open(filepath) as img:
#                 pass
#         except Exception as e:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             try:
#                 request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
#                 pic = urlopen(request)
#                 print("Downloading: " + url)
#                 with open(filepath, 'wb') as file:
#                     file.write(pic.read())
#                 img = Image.open(filepath).resize((IMAGE_HEIGHT,IMAGE_WIDTH), Image.Resampling.LANCZOS)
#                 os.remove(filepath)
#                 with open(filepath, 'wb') as file:
#                     img.save(file,"JPEG")
#             except Exception as e:
#                 print("Failed to download " + url)
#                 if os.path.exists(filepath):
#                     os.remove(filepath)

# conn = sqlite3.connect('data/scraped_ships.db')
# df = pd.read_sql('SELECT id, image_ship_url FROM scraped_ships',conn)
# df_list = np.array_split(df, N_WORKERS)

# downloader = Downloader()
# with Pool(N_WORKERS) as p:
#     p.map(downloader.download, df_list)


### Remove entries from database 

# conn = sqlite3.connect('data/scraped_ships.db')
# cur = conn.cursor()
# df = pd.read_sql('SELECT id, image_ship_url FROM scraped_ships',conn)
# print(df.shape)
# for i, row in df.iterrows():
#     id, url = row['id'], row['image_ship_url']
#     filepath = os.path.join(DATA_DIR, str(id)+'.jpg')
#     if not os.path.exists(filepath):
#         cur.execute('DELETE FROM scraped_ships WHERE id=?', (id,))
#         conn.commit()
# df = pd.read_sql('SELECT id, image_ship_url FROM scraped_ships',conn)
# print(df.shape)