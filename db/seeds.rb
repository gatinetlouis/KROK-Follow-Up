require 'csv'
require "open-uri"

puts "#{User.count} users"
puts "#{Rating.count} ratings"
puts "#{LikedRecipe.count} liked_recipe"
puts "#{Planner.count} planner"
puts "#{Recipe.count} recipe"
puts "#{PlannerRecipe.count} planner_recipe"
puts "#{ShoppingList.count} shopping_list"
puts "#{Ingredient.count} ingredient"

puts "DELETING"

Rating.destroy_all
LikedRecipe.destroy_all
Planner.destroy_all
Ingredient.destroy_all
Recipe.destroy_all
PlannerRecipe.destroy_all
ShoppingList.destroy_all
User.destroy_all
Preference.destroy_all

puts "#{User.count} users"
puts "#{Rating.count} ratings"
puts "#{LikedRecipe.count} liked_recipe"
puts "#{Planner.count} planner"
puts "#{Recipe.count} recipe"
puts "#{PlannerRecipe.count} planner_recipe"
puts "#{ShoppingList.count} shopping_list"
puts "#{Ingredient.count} ingredient"


puts "SEEDING"

cyril = User.create!(
  username: "Cyril Lignac",
  password: "azerty",
  email:    "cyril@me.com",
)

photo_c = URI.open('https://www.parisinfo.com/var/otcp/sites/images/node_43/node_51/node_532/chef-%C3%A9toil%C3%A9-cyril-lignac-%7C-630x405-%7C-%C2%A9-arthur-delloye/12145594-1-fre-FR/Chef-%C3%A9toil%C3%A9-Cyril-Lignac-%7C-630x405-%7C-%C2%A9-Arthur-Delloye.gif')
cyril.photo.attach(io: photo_c, filename: 'photocyril.png', content_type: 'image/png')
cyril.save!


amandine = User.create!(
  username: "Mandou13",
  password: "azerty",
  email:    "amandine@me.com",
)

photo_am = URI.open('https://avatars0.githubusercontent.com/u/58696274?s=400&u=2b66c76b761ef4f8842d4618d340d8daf1f7a9ce&v=4')
amandine.photo.attach(io: photo_am, filename: 'photoamandine.png', content_type: 'image/png')
amandine.save!




maxence = User.create!(
  username: "Naicma59",
  password: "azerty",
  email:    "maxence@me.com",
)

photo_m = URI.open('https://avatars3.githubusercontent.com/u/58698893?v=4')
maxence.photo.attach(io: photo_m, filename: 'photogithub.png', content_type: 'image/png')
maxence.save!

louis = User.create!(
  username: "Gat_food_lover",
  password: "azerty",
  email:    "gat@me.com",
)

photo = URI.open('https://avatars2.githubusercontent.com/u/56741514?v=4')
louis.photo.attach(io: photo, filename: "photogithub.png", content_type: 'image/png')
louis.save!

pauline = User.create!(
  username: "Paopao",
  password: "azerty",
  email:    "pauline@me.com",
)

photo_p = URI.open("https://avatars1.githubusercontent.com/u/58698738?v=4")
pauline.photo.attach(io: photo_p, filename: "photogithub.png", content_type: "image/png")
pauline.save


# first_recipe = Recipe.create!(
#   name:        "Veal Blanquette",
#   description: "From Gordon Ramsay, a blanquette is a stew made with pale-coloured meats which are not fried first",
#   steps:       "1_ Pour the stock and wine into a large saucepan, then add all the vegetables except the mushrooms. Add the garlic and rosemary and bring to the boil, then simmer, uncovered, for 10 mins.
#                 2_ Trim veal of any excess fat and cut into large bite-size cubes. Add to pan with the mushrooms. Return to a simmer, season well, then cook for 20 mins until meat is just tender; allow longer if it’s not. The stew can be cooled and chilled or frozen at this stage.
#                 3_ When ready to serve, beat together the yolks and cream. Return the stew to a simmer, stir in the horseradish or mustard, briskly mix in yolks and cream, then stir until it starts to thicken slightly. Take care not to overheat or the mixture will curdle. Stir in the butter, remove pan from the heat, then mix in the parsley, lemon zest and juice. Check the seasoning again.
#                 4_ Meanwhile, boil tagliatelle according to pack instructions – about 3 mins for fresh pasta, up to 10 mins for dried. Drain and toss with a little butter. Make into a ‘barrel’ of pasta (see above right) or simply divide between six warmed plates. Spoon the Blanquette of veal on top and serve.
#                 5_ Make a 'barrel' of tagliatelle Gordon's way: Pick up a few strands of pasta, digging a carving fork into the pot 3-4 times. Lift up the pasta so the strands hang free and press against the side of the pan. Then, holding with your (clean) fingers, start to twist the pasta into a barrel shape, pushing back against the pan side once or twice to neaten. When all the strands are wound, up-end the fork intothe centre of a plate and gently push the pasta off.",
#   dish_type:   "Main Course",
#   dish_origin: "French",
#   difficulty:  "Medium",
#   prep_time:   30,
#   servings:    6,
#   user_id:     cyril.id,
# )

# photo_first_recipe = URI.open("https://assets.afcdn.com/recipe/20190529/93189_w1024h768c1cx2886cy4330.jpg")
# first_recipe.photo.attach(io: photo_first_recipe, filename: "photo_blanquette.png", content_type: "image/png")
# first_recipe.save!

# Ingredient.create(name: 'Veal', unit: 'kg', quantity: 1, recipe_id: first_recipe.id)
# Ingredient.create(name: 'White Wine', unit: 'ml', quantity: 250, recipe_id: first_recipe.id)
# Ingredient.create(name: 'Leeks', quantity: 2, recipe_id: first_recipe.id)
# Ingredient.create(name: 'Tagliatelle', unit: 'g', quantity: 300, recipe_id: first_recipe.id)
# Ingredient.create(name: 'Carrot', quantity: 3, recipe_id: first_recipe.id)

csv_options = { col_sep: ';', quote_char: '"', headers: :first_row }

CSV.foreach(Rails.root.join('db', 'datas', 'recipes.csv'), csv_options) do |row|
  recipe = Recipe.create!(
      name: row["name"],
      global_rating: row["global_rating"],
      number_of_ratings: row["number_of_ratings"],
      steps: row["steps"],
      dish_type: row["meal_type"],
      dish_origin: row["origin"],
      prep_time: row["prep_time"],
      servings: row["servings"],
      description: row["description"],
      difficulty: row["difficulty"],
      user_id: cyril.id
      )
  photo_recipe = URI.open('https://' + row["photo_url"])
  recipe.photo.attach(io: photo_recipe, filename: "photo_#{row["name"]}.png", content_type: 'image/png')
  recipe.save!
  uuid_recipe = row["uuid"]
  CSV.foreach(Rails.root.join('db', 'datas', 'ingredients.csv'), csv_options) do |row|
    if uuid_recipe == row["uuid"]
      ingredient = Ingredient.create!(
        recipe_id: recipe.id,
        quantity: row["quantity"],
        unit: row["unit"],
        name: row["name"],
        category: row["category"],
        )
    end
  end
end



#################################
# second_recipe = Recipe.create!(
#   name:        "Veal Blanquette 2",
#   description: "desc: From Gordon Ramsay, a blanquette is a stew made with pale-coloured meats which are not fried first",
#   steps:       "1_ Pour the stock and wine into a large saucepan, then add all the vegetables except the mushrooms. Add the garlic and rosemary and bring to the boil, then simmer, uncovered, for 10 mins.
#                 2_ Trim veal of any excess fat and cut into large bite-size cubes. Add to pan with the mushrooms. Return to a simmer, season well, then cook for 20 mins until meat is just tender; allow longer if it’s not. The stew can be cooled and chilled or frozen at this stage.
#                 3_ When ready to serve, beat together the yolks and cream. Return the stew to a simmer, stir in the horseradish or mustard, briskly mix in yolks and cream, then stir until it starts to thicken slightly. Take care not to overheat or the mixture will curdle. Stir in the butter, remove pan from the heat, then mix in the parsley, lemon zest and juice. Check the seasoning again.
#                 4_ Meanwhile, boil tagliatelle according to pack instructions – about 3 mins for fresh pasta, up to 10 mins for dried. Drain and toss with a little butter. Make into a ‘barrel’ of pasta (see above right) or simply divide between six warmed plates. Spoon the Blanquette of veal on top and serve.
#                 5_ Make a 'barrel' of tagliatelle Gordon's way: Pick up a few strands of pasta, digging a carving fork into the pot 3-4 times. Lift up the pasta so the strands hang free and press against the side of the pan. Then, holding with your (clean) fingers, start to twist the pasta into a barrel shape, pushing back against the pan side once or twice to neaten. When all the strands are wound, up-end the fork intothe centre of a plate and gently push the pasta off.",
#   dish_type:   "main",
#   dish_origin: "French",
#   difficulty:  "Medium",
#   prep_time:   30,
#   servings:    6,
#   user_id:     cyril.id,
# )

# photo_second_recipe = URI.open("https://static.750g.com/images/622-auto/d60923a6a32c8b20cfc5d1f7db28b97d/blanquette-de-veau.jpg")
# second_recipe.photo.attach(io: photo_second_recipe, filename: "photo_blanquette2.png", content_type: "image/png")
# second_recipe.save!

# Ingredient.create(name: 'Veal', unit: 'kg', quantity: 1, recipe_id: second_recipe.id)
# Ingredient.create(name: 'White Wine', unit: 'ml', quantity: 250, recipe_id: second_recipe.id)
# Ingredient.create(name: 'Leeks', quantity: 2, recipe_id: second_recipe.id)
# Ingredient.create(name: 'Tagliatelle', unit: 'g', quantity: 300, recipe_id: second_recipe.id)
# Ingredient.create(name: 'Carrot', quantity: 3, recipe_id: second_recipe.id)

Planner.create(
  name: "#{Time.now.strftime("%d")}/#{Time.now.strftime("%m")} Menu",
  user_id: maxence.id
)

Planner.create(
  name: "#{Time.now.strftime("%d")}/#{Time.now.strftime("%m")} Menu",
  user_id: louis.id
)

Planner.create(
  name: "#{Time.now.strftime("%d")}/#{Time.now.strftime("%m")} Menu",
  user_id: pauline.id
)

Planner.create(
  name: "#{Time.now.strftime("%d")}/#{Time.now.strftime("%m")} Menu",
  user_id: amandine.id
)

# PlannerRecipe.create(
#   recipe_id: Recipe.first.id,
#   planner_id: louis.planners.last.id,
#   servings: Recipe.first.servings,
#   cooked: false
#   )

# PlannerRecipe.create(
#   recipe_id: Recipe.first.id,
#   planner_id: pauline.planners.last.id,
#   servings: Recipe.first.servings,
#   cooked: false
#   )

# PlannerRecipe.create(
#   recipe_id: Recipe.first.id,
#   planner_id: maxence.planners.last.id,
#   servings: Recipe.first.servings,
#   cooked: false
#   )

# PlannerRecipe.create(
#   recipe_id: Recipe.second.id,
#   planner_id: louis.planners.last.id,
#   servings: Recipe.second.servings,
#   cooked: false
#   )

# PlannerRecipe.create(
#   recipe_id: Recipe.second.id,
#   planner_id: pauline.planners.last.id,
#   servings: Recipe.second.servings,
#   cooked: false
#   )

# PlannerRecipe.create(
#   recipe_id: Recipe.second.id,
#   planner_id: maxence.planners.last.id,
#   servings: Recipe.second.servings,
#   cooked: false
#   )

Rating.create(
  user_id: louis.id,
  recipe_id: Recipe.first.id,
  rating: 3
  )
#############################################




puts "#{User.count} users"
puts "#{Rating.count} rating"
puts "#{LikedRecipe.count} liked_recipe"
puts "#{Planner.count} planner"
puts "#{Recipe.count} recipe"
puts "#{PlannerRecipe.count} planner_recipe"
puts "#{ShoppingList.count} shopping_list"
puts "#{Ingredient.count} ingredient"


