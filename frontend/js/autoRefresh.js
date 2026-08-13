document.addEventListener("DOMContentLoaded", () => {

    // Initial page loading happens normally

    setInterval(() => {

        console.log("Auto refreshing dashboard...");

        location.reload();

    }, 1800000);

});