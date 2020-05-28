
const update_servings = () => {

const plus = document.getElementById('btn-plus-recipe');

  if (plus) {
    const minus = document.getElementById('btn-minus-recipe');
    let servings = document.getElementById('recipe-servings');
    const ingredient_quantities = document.querySelectorAll('.ingredient-quantity');

    minus.addEventListener('click', (ev) => {
      ev.preventDefault()
      let servings_to_update = parseInt(servings.innerHTML, 10)
      servings.innerText = servings_to_update - 1;
      ingredient_quantities.forEach((quantity) => {
        let ingredient_per_pers = quantity.dataset.quantity;
        quantity.innerHTML = ingredient_per_pers * (servings_to_update - 1);
      });
    });

    plus.addEventListener('click', (ev) => {
      ev.preventDefault()
      let servings_to_update = parseInt(servings.innerHTML, 10)
      servings.innerText = servings_to_update + 1;
      ingredient_quantities.forEach((quantity) => {
        let ingredient_per_pers = quantity.dataset.quantity;
        quantity.innerHTML = ingredient_per_pers * (servings_to_update + 1);
      });
    });
  }
}

export { update_servings }
