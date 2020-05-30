import sys
import pandas as pd
import numpy as np
import collections
# from gensim.models import word2vec
from gensim.models import Word2Vec
from scipy import spatial

current_recipe_id = sys.argv[1]
current_recipe_name = sys.argv[2]


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
model = Word2Vec.load("word2vec.model")
# model = Word2Vec(min_count=0,
#                      window=10,
#                      size=20,
#                      sample=6e-5,
#                      alpha=0.3,
#                      min_alpha=0.0007,
#                      workers=4,
#                     sg = 1)

# model.build_vocab(list_of_ingredients, progress_per=10)
# model.train(list_of_ingredients, total_examples=model.corpus_count, epochs=500, report_delay=1)
# model.init_sims(replace=True)
# model.save("word2vec.model")
# print(model.wv.most_similar('chocolate'))
# print(list_of_ingredients)
# print(ingredients_df.shape)
# print(recipes_df.shape)
index2word_set = set(model.wv.index2word)

def avg_feature_vector(recipe_array, model, num_features, index2word_set):
    ingredients = list(set(recipe_array))
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_ingredients = 0
    for ingredient in ingredients:
        if ingredient in index2word_set:
            n_ingredients += 1
            feature_vec = np.add(feature_vec, model.wv[ingredient])
    if (n_ingredients > 0):
        feature_vec = np.divide(feature_vec, n_ingredients)
    return feature_vec

def recommended_recipe(ingredients_for_one_recipe, ingredients_for_all_recipes):

    afv_list = list()
    similarity_score = list()
    afv_recipe = avg_feature_vector(ingredients_for_one_recipe, model=model, num_features=20, index2word_set=index2word_set)

    for i in range(len(ingredients_for_all_recipes)):
        afv = avg_feature_vector(ingredients_for_all_recipes[i], model=model, num_features=20, index2word_set=index2word_set)
        afv_list.append(afv)
        sim = 1 - spatial.distance.cosine(afv_recipe, afv_list[i])
        similarity_score.append(sim)

    index = np.argsort(similarity_score)[-2]
    return list_of_ingredients[index], index

def recommended_recipe_second(ingredients_for_one_recipe, ingredients_for_all_recipes):

    afv_list = list()
    similarity_score = list()
    afv_recipe = avg_feature_vector(ingredients_for_one_recipe, model=model, num_features=20, index2word_set=index2word_set)

    for i in range(len(ingredients_for_all_recipes)):
        afv = avg_feature_vector(ingredients_for_all_recipes[i], model=model, num_features=20, index2word_set=index2word_set)
        afv_list.append(afv)
        sim = 1 - spatial.distance.cosine(afv_recipe, afv_list[i])
        similarity_score.append(sim)

    index = np.argsort(similarity_score)[-3]
    return list_of_ingredients[index], index

def find_index_with_recipe_name(recipe_name):
    true_false = recipes_df_joined[["name"]] == recipe_name
    index = np.where(true_false)[0][0]
    return index

def recommender(recipe_name):
    index = find_index_with_recipe_name(recipe_name)
    ingredients_recipe = list_of_ingredients[index]
    ingredients_reco, index_new = recommended_recipe(ingredients_recipe, list_of_ingredients)
    ingredients_reco_second, index_new_second = recommended_recipe_second(ingredients_recipe, list_of_ingredients)
    recipe = recipes_df_joined.iloc[index,3]
    reco = recipes_df_joined.iloc[index_new,3]
    reco_second = recipes_df_joined.iloc[index_new_second,3]
    both_recos = [reco,reco_second]
    #print("current recipe:", recipe)
    #print("current ingredients:", ingredients_recipe)
    #print("                   ")
    #print("recommended recipe:", reco)
    #print("recommended ingredients:", ingredients_reco)
    return both_recos

print(recommender(current_recipe_name))


