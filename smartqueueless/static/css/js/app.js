document.addEventListener("DOMContentLoaded", function () {

    const flashes = document.querySelectorAll(".flash");

    flashes.forEach(function (flash) {

        setTimeout(function () {

            flash.style.opacity = "0";
            flash.style.transform = "translateX(20px)";
            flash.style.transition = "0.4s";

            setTimeout(function () {
                flash.remove();
            }, 400);

        }, 4000);

    });

});