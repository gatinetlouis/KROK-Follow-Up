Rails.application.routes.draw do
  get 'shopping_lists/index'
  get 'shopping_lists/update'
  devise_for :users

  # root to: 'recipes#index'
  resources :supermarckets, only: :index

  resources :testhere, only: :index

  root to: 'pages#home'


  resource :profile, only: :show

  resources :recipes do
    resources :planner_recipes, only: [:create, :destroy]
    resources :liked_recipes, only: :create
  end

  resources :planner_recipes, only: :show do
    resources :ratings, only: :create
  end

  resources :planners, only: [:show, :create, :update] do
    member do
      post 'add_from_previous_planner'
    end
    resources :planner_recipes, only: [:destroy, :update]
    resources :shopping_lists, only: [:index, :update]
  end
  resources :liked_recipes, only: [:index, :destroy]
  resources :preferences, only: [:create, :destroy]
end
