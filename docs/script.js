const slider = document.querySelector("#comparison-slider");
const comparison = document.querySelector(".comparison");

slider.addEventListener("input", (event) => {
  comparison.style.setProperty("--position", `${event.target.value}%`);
});

