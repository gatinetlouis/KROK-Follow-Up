import sys
import pandas as pd
import numpy as np
import collections
# from gensim.models import word2vec
# from gensim.models import Word2Vec
# from scipy import spatial

arg_1 = sys.argv[1]
arg_2 = sys.argv[2]

print("arg1:", arg_1)
print("arg2:", arg_2)
ingredients_df = pd.read_csv('/home/louis/code/gatinetlouis/KROK-Follow-Up/db/datas/ingredients.csv', sep=";")
recipes_df = pd.read_csv('/home/louis/code/gatinetlouis/KROK-Follow-Up/db/datas/recipes.csv',sep=";")
print(ingredients_df.shape)
print(recipes_df.shape)
