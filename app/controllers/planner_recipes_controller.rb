class PlannerRecipesController < ApplicationController

  def create
    @planner_recipe = PlannerRecipe.new(cooked: false, servings: 1)
    @planner = current_user.planners.last
    @planner_recipe.planner = @planner
    @recipe = Recipe.find(params[:recipe_id])
    @planner_recipe.recipe = @recipe
    @planner_recipe.save!

    @recipe.ingredients.each do |ingredient|

      shopping_lists = ShoppingList.where("planner_id = ? AND ingredient_name = ?", @planner.id, ingredient.name)

      quotient = @planner_recipe.servings.to_i.fdiv(@planner_recipe.recipe.servings.to_i)
      ingredient.quantity = (quotient * ingredient.quantity)
      ingredient = ingredient.convert_ingredient_to_krok_unit unless ingredient.krok_unit?

      if shopping_lists.empty?

        ShoppingList.create!(
          ingredient_name: ingredient.name,
          quantity:        ingredient.quantity,
          unit:            ingredient.unit,
          bought:          false,
          planner:         @planner,
          category:        ingredient.category
        )
      else
        shopping_list_reference = shopping_lists.first
        shopping_list_reference.quantity += ingredient.quantity
        shopping_list_reference.save!
      end

    end

    if URI(request.referer).path.index("planners") == 1
      redirect_to planner_path(@planner)
    else
      redirect_to recipes_path
    end
  end

  def update
    @planner = Planner.find(params[:planner_id])
    @planner_recipe = PlannerRecipe.find(params[:id])
    @planner_recipe.cooked = !@planner_recipe.cooked
    @planner_recipe.save
    redirect_to planner_path(@planner)
  end

  def destroy
    @planner = params[:planner_id].nil? ? current_user.planners.last : Planner.find(params[:planner_id])
    @planner_recipe = PlannerRecipe.find(params[:id])
    @recipe = Recipe.find(@planner_recipe.recipe_id)

    @recipe.ingredients.each do |ingredient|

      shopping_list = ShoppingList.where("planner_id = ? AND ingredient_name = ?", @planner.id, ingredient.name)


      quotient = @planner_recipe.servings.to_i.fdiv(@planner_recipe.recipe.servings.to_i)
      ingredient.quantity = (quotient * ingredient.quantity)
      ingredient = ingredient.convert_ingredient_to_krok_unit unless ingredient.krok_unit?
      if !shopping_list.empty?
        shopping_list.first.quantity -= ingredient.quantity
        if shopping_list.first.quantity <= 0
          shopping_list.first.destroy!
        else
          shopping_list.first.save!
        end
      end
    end

    @planner_recipe.destroy

    if URI(request.referer).path.index("planners") == 1
      redirect_to planner_path(@planner)
    else
      redirect_to recipes_path
    end
  end

  private

  def set_planner_recipe_params
    params.require(:planner_recipe).permit(:cooked, :servings)
  end
end
