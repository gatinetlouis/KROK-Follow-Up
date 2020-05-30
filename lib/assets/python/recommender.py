import sys
import pandas as pd
import numpy as np
import collections
# from gensim.models import word2vec
# from gensim.models import Word2Vec
from scipy import spatial

arg_1 = sys.argv[1]
arg_2 = sys.argv[2]

print("arg1:", arg_1)
print("arg2:", arg_2)
ingredients_df = pd.read_csv('/home/louis/code/gatinetlouis/KROK-Follow-Up/db/datas/ingredients.csv', sep=";")
recipes_df = pd.read_csv('/home/louis/code/gatinetlouis/KROK-Follow-Up/db/datas/recipes_new.csv',sep=";")
ingredients_by_recipe = ingredients_df[["uuid", "name"]].groupby(['uuid']).agg({'name':lambda x: list(x)})
ingredients_by_recipe["ingredient_category_list"] = ingredients_df[["uuid", "category"]].groupby(['uuid']).agg({'category': lambda y: list(y)})
ingredients_by_recipe["num_of_ingredients"] = ingredients_df[["uuid", "name"]].groupby(['uuid']).size()
ingredients_by_recipe.columns = ["ingredient_list","ingredient_category_list","num_of_ingredients"]
recipes_df_joined = pd.merge(recipes_df, ingredients_by_recipe, on='uuid')
nb_of_recipes = recipes_df_joined.shape[0]
list_of_ingredients = list()
for i in range(nb_of_recipes):
    list_of_ingredients.append(list(set(recipes_df_joined.iloc[i,-3]))) #to add unique list of ingredients
print(list_of_ingredients)
# print(ingredients_df.shape)
# print(recipes_df.shape)
