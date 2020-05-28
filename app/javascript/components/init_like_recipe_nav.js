
const initLikedRecipeNav = () => {
  const btnStarter = document.getElementById('btn-starter');
  const btnMain = document.getElementById('btn-main');
  const btnDessert = document.getElementById('btn-dessert');

  const mainRecipes = document.querySelector('.liked-recipes-main');
  const starterRecipes = document.querySelector('.liked-recipes-starter');
  const dessertRecipes = document.querySelector('.liked-recipes-dessert');


  if (btnMain) {
    btnMain.addEventListener('click', (event) => {
      btnStarter.classList.remove('btn-liked-active');
      btnDessert.classList.remove('btn-liked-active');
      btnMain.classList.add('btn-liked-active');
      mainRecipes.classList.remove('d-none');
      starterRecipes.classList.add('d-none');
      dessertRecipes.classList.add('d-none');
    })
  }
  if (btnStarter) {
    btnStarter.addEventListener('click', (event) => {
      btnMain.classList.remove('btn-liked-active');
      btnDessert.classList.remove('btn-liked-active');
      btnStarter.classList.add('btn-liked-active');
      starterRecipes.classList.remove('d-none');
      mainRecipes.classList.add('d-none');
      dessertRecipes.classList.add('d-none');
    })
  }
  if (btnDessert) {
    btnDessert.addEventListener('click', (event) => {
      btnMain.classList.remove('btn-liked-active');
      btnStarter.classList.remove('btn-liked-active');
      btnDessert.classList.add('btn-liked-active');
      dessertRecipes.classList.remove('d-none');
      starterRecipes.classList.add('d-none');
      mainRecipes.classList.add('d-none');
    })
  }

}

export { initLikedRecipeNav };
