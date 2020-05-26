class Recipe < ApplicationRecord

  DISH_TYPE = ["dessert", "main", "starter"]
  PREP_TIME = ["20", "30", "45", "60"]
  DISH_ORIGIN = ["Italian", "German", "French", "American", "English", "Middle East", "British", "Mexican", "Spanish", "Indian"]
  DIFFICULTY = ["Easy", "Medium","Hard"]

  belongs_to :user
  has_many   :ingredients
  has_many   :planner_recipes
  has_many   :liked_recipes
  has_many   :ratings

  validates  :name, presence: true
  validates  :description, presence: true, uniqueness: true
  validates  :dish_type, inclusion: { in: ['dessert', 'starter', 'main'] }
  validates  :dish_origin, inclusion: { in: ['Italian', 'Mexican', 'British', 'Indian', 'French', 'Greek', 'English', 'Chinese',  'Middle East', 'Thai', 'Vietnamese', 'Spanish',  'Asian', 'Korean', 'American', 'German'] }
  validates  :prep_time, presence: true
  validates  :servings, presence: true
  validates  :difficulty, inclusion: { in: ['Easy', 'Medium', 'Hard'] }


  has_one_attached :photo

  include PgSearch::Model
  pg_search_scope :search_by_name_origin_dish_type_and_difficulty,
    against: [ :name, :dish_origin, :dish_type, :difficulty ],
    using: {
      tsearch: { prefix: true }
    }


  def avg_rating
    sum = 0
    self.ratings.each do |rating|
      sum += rating.rating
    end
    self.ratings.size == 0 ? 0 : (sum / self.ratings.size)
  end

  def liked?(user)
    answer = []
    self.liked_recipes.each do |liked_recipe|
      answer << (liked_recipe.user.id == user.id)
    end
    answer.include?(true)
  end

  def added?(user)
    answer = []
    user.planners.last.planner_recipes.each do |planner_recipe|
      answer << (planner_recipe.recipe == self)
    end
    answer.include?(true)
  end

  def already_rated?(user)
    user.ratings.each do |rate|
      return true if rate.recipe == self
    end
    return false
  end

  def categories_of_recipe
    categories = []
    self.ingredients.each do |ingredient|
      categories << ingredient.category
    end
    return categories.uniq
  end

  def optim_prep_time
    if self.prep_time < 60
      "#{self.prep_time} m"
    else
      hour = self.prep_time / 60
      minutes = self.prep_time.remainder(60)
      "#{hour}h #{minutes}m"
    end
  end

  def is_rejected?(user)
    unless user.nil?
      self.ingredients.each do |ingredient|
        # return true if user.ingredients.include?(ingredient.name)
        user.ingredients.each do |user_ingredient|
          return true if user_ingredient.name == ingredient.name
        end
      end
      return false
    end
  end
end
