

class TesthereController < ApplicationController

  def index
    @result = ScrappingRecipes.scrapping('starter', 'english')
  end
end
