import pandas as pd
import numpy as np
data= pd.read_csv("../series_names_data/netflix_titles.csv", encoding="utf-8")

list_data = data['title'][:1000]

print(list_data, 'list data')