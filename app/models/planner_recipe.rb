class PlannerRecipe < ApplicationRecord
  belongs_to :planner
  belongs_to :recipe
  validates  :servings, presence: true, numericality: { greater_than_or_equal_to: 1 }


  def revert_cook_status
    if self.cooked
      self.cooked = false
      self.save
    else
      self.cooked = true
      self.save
    end
  end

  def repeatedly_created?(planner)
    answer = []
    planner.planner_recipes.each { |pr| answer << (pr.recipe.id == self.recipe.id) }
    answer.include?(true)
  end
end
