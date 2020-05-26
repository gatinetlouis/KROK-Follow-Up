class AddNumberOfRatingsToRecipe < ActiveRecord::Migration[5.2]
  def change
    add_column :recipes, :number_of_ratings, :integer
  end
end
