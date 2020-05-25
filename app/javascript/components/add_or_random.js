
const display_add_or_random = () => {

const cards_to_display = document.querySelectorAll('#card_to_display')
const cards_to_hide = document.querySelectorAll('#card_to_hide')
const cards_update_launch = document.querySelectorAll('#card-img-icon-add')
const back_to_cards_through_search = document.querySelectorAll('#card-img-icon-search')
const back_to_cards_through_random = document.querySelectorAll('#card-img-icon-random')
const cards_img_icon_random_valid = document.querySelectorAll('#card-img-icon-random-valid')
const cards_img_icon_random_again = document.querySelectorAll('#card-img-icon-random-again')
const card_random_to_hide_starter = document.querySelectorAll('.card_random_to_hide-0')
const card_random_to_hide_main = document.querySelectorAll('.card_random_to_hide-1')
const card_random_to_hide_dessert = document.querySelectorAll('.card_random_to_hide-2')


const remove_d_none_for_random_card = (array) => {
  const card = array[Math.floor(Math.random() * array.length)]
  card.classList.remove("d-none")
  card.classList.add("active_random")
  card.querySelector(".card-img-icon-random-again").addEventListener('click', (event) => {
    put_d_none_for_active_card(array);
    remove_d_none_for_random_card(array);
   })
  }


const put_d_none_for_active_card = (array) => {
  array.forEach(element => {
    if (element.classList.contains("active_random")) {
      element.classList.remove(".active_random")
      element.classList.add("d-none")
    }
  })
}


if (
  cards_to_display.length !==0 &&
  cards_to_hide.length !==0 &&
  cards_update_launch.length !==0 &&
  back_to_cards_through_search.length !==0 &&
  back_to_cards_through_random.length !==0 &&
  cards_img_icon_random_valid.length !==0 &&
  cards_img_icon_random_again.length !==0 &&
  card_random_to_hide_starter.length !==0 &&
  card_random_to_hide_main.length !==0 &&
  card_random_to_hide_dessert.length !==0
  ) {
    [0,1,2].forEach(index => {
      cards_update_launch[index].addEventListener('click', (event) => {
        cards_to_display[index].classList.remove("d-none");
        cards_to_hide[index].classList.add("d-none");
      });

      back_to_cards_through_search[index].addEventListener('click', (event) => {
        cards_to_display[index].classList.add("d-none");
        cards_to_hide[index].classList.remove("d-none");
      });

      back_to_cards_through_random[index].addEventListener('click', (event) => {
        cards_to_display[index].classList.add("d-none");
        cards_to_hide[index].classList.remove("d-none");
        if (index === 0) remove_d_none_for_random_card(card_random_to_hide_starter);
        if (index === 1) remove_d_none_for_random_card(card_random_to_hide_main);
        if (index === 2) remove_d_none_for_random_card(card_random_to_hide_dessert);
      });
    });
  };
};


export { display_add_or_random }
