class RecipesController < ApplicationController
  skip_before_action :authenticate_user!, only: [:index, :show]
  before_action :find_recipe, only: [:show, :edit, :update]

  def index
    @fake_home = params[:home]

    @recipes = Recipe.all
    @planner = current_user.planners.last unless current_user.nil?
    # @recipes = Recipe.all.shuffle.first(30)
    if params[:dish_type].present?
      @recipes = Recipe.where(dish_type: params[:dish_type])
    elsif params[:dish_origin].present?
       @recipes = Recipe.where(dish_origin: params[:dish_origin])
    elsif params[:difficulty].present?
      @recipes = Recipe.where(difficulty: params[:difficulty])
    elsif params[:prep_time].present?
      params[:prep_time] = params[:prep_time].to_i
      @recipes =Recipe.where("prep_time <= ?", params[:prep_time])
    elsif params[:query].present?
      @recipes = []
      if Recipe.search_by_name_origin_dish_type_and_difficulty(params[:query]).empty? == false
        @recipes << Recipe.search_by_name_origin_dish_type_and_difficulty(params[:query]).to_a
        @recipes = @recipes.first
      end
      if Ingredient.search_by_ingredient(params[:query]).empty? == false
        Ingredient.search_by_ingredient(params[:query]).each do |ingredient|
          @recipes << Recipe.find(ingredient.recipe_id)
        end
      end
    end
    @recipes = @recipes.uniq
    if  (params[:query_dish_type] || params[:query_prep_time] || params[:query_dish_origin] || params[:query_difficulty]).present?
      dish_type = params[:query_dish_type].present? ? params[:query_dish_type].downcase.split("%") : Recipe::DISH_TYPE
      prep_time = params[:query_prep_time].present? ? params[:query_prep_time].downcase.split("%").map {|time| time[0...-3].to_i}.max : 200
      dish_origin = params[:query_dish_origin].present? ? params[:query_dish_origin].split("%") : Recipe::DISH_ORIGIN
      difficulty = params[:query_difficulty].present? ? params[:query_difficulty].split("%") : Recipe::DIFFICULTY
      @recipes = Recipe.where(dish_type: dish_type, dish_origin: dish_origin, difficulty: difficulty).where("prep_time <= #{prep_time}")
    end
  end

  def show
    @recipe_id = @recipe.id
    @python_result = `python lib/assets/python/recommender.py "#{@recipe_id}" "#{@recipe.name}"`
  end

  def new
    @recipe = Recipe.new
  end

  def create
    @recipe = Recipe.new(recipe_params)
    @recipe.user = current_user
    if @recipe.save
      redirect_to recipe_path(@recipe)
    else
      render :new
    end
  end

  def edit
  end

  def update
    if @recipe.update(recipe_params)
      redirect_to recipe_path(@recipe)
    else
      render :edit
    end
  end

  def destroy
  end

  private

  def recipe_params
    params.require(:recipe).permit(:name, :description, :dish_type, :dish_origin, :prep_time, :servings)
  end

  def find_recipe
    @recipe = Recipe.find(params[:id])
  end
end
