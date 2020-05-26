require 'nokogiri'
require 'open-uri'
require 'csv'

# ScrappingRecipes.scrapping('starter', '')
# ScrappingRecipes.scrapping('main', '')
# ScrappingRecipes.scrapping('dessert', '')

class ScrappingRecipes
  URL_BBC = 'https://www.bbcgoodfood.com/'



  def self.testing
    url_test = "https://www.bbcgoodfood.com/recipes/mexican-sweet-potato-soup"
    recipe_page = Nokogiri::HTML(open(url_test), nil, 'utf-8')
    global_rating =  recipe_page.search(".rate-fivestar-btn-filled").size
    number_of_ratings = recipe_page.search(".rate-info").text.split(' ').first[1..-1]
    "global rating : " + global_rating.to_s + "and number of ratings : " + number_of_ratings.to_s

  end



  def self.scrapping(meal_type, meal_origin)
    csv_options = { col_sep: ';', force_quotes: true, quote_char: '"' }
    url_search  = URL_BBC + 'search/recipes?query='  + meal_type + ' ' +meal_origin
    doc         = Nokogiri::HTML(open(url_search), nil, 'utf-8')
    names       = scrapping_search(doc, ".teaser-item__title")
    prep_times  = cooking_time(scrapping_search(doc, ".teaser-item__info-item--total-time"))
    difficulty  = scrapping_search(doc, ".teaser-item__info-item--skill-level")
    description = scrapping_search(doc, '.even')
    url_meals   = scrapping_h3_href(doc, 'teaser-items')

    recipes = []
    ingredients = []
    photo_urls = []

    names.each_with_index do |recipe, index|
      recipe_page = Nokogiri::HTML(open(url_meals[index]), nil, 'utf-8')
      servings = recipe_page.at("//span[@itemprop = 'recipeYield']").children.text.scan(/\d/)[0]
      global_rating =  recipe_page.search(".rate-fivestar-btn-filled").size
      number_of_ratings = recipe_page.search(".rate-info").text.split(' ').first[1..-1]
      steps = scrapping_search(recipe_page, ".method__list")
      photo_urls << recipe_page.xpath("//img[@itemprop = 'image']").map { |t| t[:src].remove(/\/{2}/) }.first
      uuid = SecureRandom.hex(10)
      recipes << [uuid, global_rating, number_of_ratings, names[index], steps.first, meal_type, meal_origin, prep_times[index], servings, description[index], difficulty[index], photo_urls[index]]

      scrapping_search(recipe_page, ".ingredients-list__item").each do |ingredient|
        quantity = ingredient.split(" ")[0].scan(/\d/).join.to_i
        unit = ingredient.split(" ")[0].scan(/\D/).join
        unit = 'tbsp' if unit == ""
        name = ingredient.split(" ").drop(1).join(" ")
        ingredients << [uuid, quantity, unit, name]
        p ingredients.first(5)
      end
    end

    photo_urls = []
    names.each_with_index do |recipe, index|
      recipe_page = Nokogiri::HTML(open(url_meals[index]), nil, 'utf-8')
      photo_urls = recipe_page.xpath("//img[@itemprop = 'image']").map { |t| t[:src].remove(/\/{2}/) }.first
      p photo_urls
    end

    CSV.open(Rails.root.join('db', 'datas', 'ingredients.csv'), 'ab', csv_options) do |csv|
      csv << ['uuid', 'quantity', 'unit', 'name']
      ingredients.each do |ingredienty|
        csv  << ingredienty
      end
    end

    CSV.open(Rails.root.join('db', 'datas', 'recipes.csv'), 'ab', csv_options) do |csv|
      csv << ['uuid', 'global_rating', 'number_of_ratings', 'name', 'steps', 'meal_type', 'origin', 'prep_time', 'servings', 'description', 'difficulty', 'photo_url']
      recipes.each do |recipe|
        csv << recipe
      end
    end

    "Done"
  end

  def self.scrapping_search(doc, tag)
    array = []
    doc.search(tag).each { |recipe| array << recipe.text.strip }
    array
  end


  def self.scrapping_h3_href(doc, tag)
    doc.xpath('//h3[@class="teaser-item__title"]/a/@href').map { |recipe| URL_BBC + recipe.value }
    # doc.search(tag).map { |recipe| recipe.value }
  end

  def self.cooking_time(cooking_time_array)
    cooking_time_array.map do |time|
      if time.include?("hour") || time.include?("hours")
        time = time.split(" ")
        hours = time[0].to_i * 60
        minutes = time.drop(1).join(" ").scan(/\d/).join('').to_i
        time = minutes + hours
      else
        time = time.split(" ")
        time = time[0].to_i
      end
    time
    end
  end

end
