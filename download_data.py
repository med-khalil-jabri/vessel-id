import os
import sqlite3
import numpy as np
import pandas as pd
from multiprocessing import Pool
from urllib.request import Request, urlopen
from PIL import Image
DATA_DIR = "data"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
N_WORKERS = 64

class Downloader():
  def download(self, df):
    for _, row in df.iterrows():
        id, url = row['id'], row['image_ship_url']
        filepath = os.path.join(DATA_DIR, str(id)+'.jpg')
        if not os.path.exists(filepath):
            try:
                request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                pic = urlopen(request)
                print("downloading: " + url)
                with open(filepath, 'wb') as file:
                    file.write(pic.read())
                img = Image.open(filepath).resize((IMAGE_HEIGHT,IMAGE_WIDTH), Image.Resampling.LANCZOS)
                os.remove(filepath)
                with open(filepath, 'wb') as file:
                    img.save(file,"JPEG")
            except Exception as e:
                print("failed to download " + url)
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            if os.path.getsize(filepath) == 0:
                os.remove(filepath)
                print("removed corrupted file :", filepath)


conn = sqlite3.connect('scraped_ships.db')
df = pd.read_sql('SELECT id, image_ship_url FROM scraped_ships',
            conn)

df_list = np.array_split(df, N_WORKERS)

downloader = Downloader()
with Pool(N_WORKERS) as p:
    p.map(downloader.download, df_list)
