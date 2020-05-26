class AddGlobalRatingToRecipe < ActiveRecord::Migration[5.2]
  def change
    add_column :recipes, :global_rating, :integer
  end
end
