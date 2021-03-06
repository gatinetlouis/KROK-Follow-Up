const display_searchbar = () => {
  const SearchBarIndex = document.getElementById('searchbox-recipe-index-before');
  if (SearchBarIndex) {
    SearchBarIndex.addEventListener("click", (event) => {
      document.getElementById('recipes').classList.add("d-none");
      document.getElementById('searchbox-recipe-index-after').classList.remove("d-none");
    });
  }
}

const display_filter = () => {
  const DisplayFilters = document.getElementById('display-filters-title');

  if (DisplayFilters) {

    DisplayFilters.addEventListener("click", (event) => {
      document.getElementById('display-filters-title').classList.toggle("d-none");
      document.getElementById('searchbox-recipe-index-before').classList.toggle("d-none");
      document.getElementById('display-filters').classList.toggle("d-none");
    });
  }
}

export { display_searchbar }
export { display_filter }
