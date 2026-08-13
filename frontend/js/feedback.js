const FEEDBACK_API_URL =
    "http://localhost:5000/api/feedback";


// ==========================================
// DOM Loaded
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        initializeFeedback();

    }
);


// ==========================================
// Initialize
// ==========================================

function initializeFeedback() {

    const modal =
        document.getElementById("feedback-modal");

    const openFeedback =
        document.getElementById(
            "open-feedback-btn"
        );

    const openSuggestion =
        document.getElementById(
            "open-suggestion-btn"
        );

    const closeButton =
        document.getElementById(
            "close-feedback-btn"
        );

    const form =
        document.getElementById(
            "feedback-form"
        );

    const message =
        document.getElementById(
            "feedback-message"
        );


    if (!modal || !form) {

        return;

    }


    // Feedback

    openFeedback?.addEventListener(
        "click",
        () => {

            openFeedbackModal("feedback");

        }
    );


    // Suggestion

    openSuggestion?.addEventListener(
        "click",
        () => {

            openFeedbackModal("suggestion");

        }
    );


    // Close

    closeButton?.addEventListener(
        "click",
        closeFeedbackModal
    );


    // Click outside modal

    modal.addEventListener(
        "click",
        (event) => {

            if (
                event.target === modal
            ) {

                closeFeedbackModal();

            }

        }
    );


    // Submit

    form.addEventListener(
        "submit",
        submitFeedback
    );


    // Character counter

    message?.addEventListener(
        "input",
        updateCharacterCount
    );

}


// ==========================================
// Open Modal
// ==========================================

function openFeedbackModal(type) {

    const modal =
        document.getElementById(
            "feedback-modal"
        );

    const typeInput =
        document.getElementById(
            "feedback-type"
        );

    const title =
        document.getElementById(
            "feedback-modal-title"
        );

    const icon =
        document.getElementById(
            "feedback-modal-icon"
        );

    const status =
        document.getElementById(
            "feedback-status"
        );


    typeInput.value = type;


    if (type === "suggestion") {

        title.textContent =
            "Suggest an Idea";

        icon.textContent = "💡";

    }

    else {

        title.textContent =
            "Give Feedback";

        icon.textContent = "💬";

    }


    status.textContent = "";


    modal.classList.add("active");


    document.body.classList.add(
        "feedback-modal-open"
    );

}


// ==========================================
// Close Modal
// ==========================================

function closeFeedbackModal() {

    const modal =
        document.getElementById(
            "feedback-modal"
        );

    modal?.classList.remove("active");

    document.body.classList.remove(
        "feedback-modal-open"
    );

}


// ==========================================
// Character Count
// ==========================================

function updateCharacterCount() {

    const message =
        document.getElementById(
            "feedback-message"
        );

    const count =
        document.getElementById(
            "feedback-count"
        );


    count.textContent =
        message.value.length;

}


// ==========================================
// Submit Feedback
// ==========================================

async function submitFeedback(event) {

    event.preventDefault();


    const token =
        localStorage.getItem("token");


    if (!token) {

        alert(
            "Please login before submitting feedback."
        );

        return;

    }


    const type =
        document.getElementById(
            "feedback-type"
        ).value;


    const subject =
        document.getElementById(
            "feedback-subject"
        ).value.trim();


    const message =
        document.getElementById(
            "feedback-message"
        ).value.trim();


    const ratingValue =
        document.getElementById(
            "feedback-rating"
        ).value;


    const status =
        document.getElementById(
            "feedback-status"
        );


    if (!message) {

        status.textContent =
            "Please enter a message.";

        status.className =
            "feedback-status error";

        return;

    }


    const rating =
        ratingValue
            ? Number(ratingValue)
            : null;


    try {

        status.textContent =
            "Submitting...";

        status.className =
            "feedback-status loading";


        const response =
            await fetch(
                FEEDBACK_API_URL,
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "Authorization":
                            `Bearer ${token}`

                    },

                    body: JSON.stringify({

                        type,

                        subject,

                        message,

                        rating

                    })

                }
            );


        const data =
            await response.json();


        if (response.status === 401) {

            localStorage.removeItem(
                "token"
            );

            window.location.href =
                "login.html";

            return;

        }


        if (!response.ok) {

            throw new Error(
                data.message ||
                "Unable to submit feedback."
            );

        }


        status.textContent =
            "✓ Thank you! Your response has been submitted.";

        status.className =
            "feedback-status success";


        document
            .getElementById(
                "feedback-form"
            )
            .reset();


        document
            .getElementById(
                "feedback-count"
            )
            .textContent = "0";


        setTimeout(
            () => {

                closeFeedbackModal();

            },
            1800
        );

    }

    catch (error) {

        console.error(
            "Feedback Error:",
            error
        );


        status.textContent =
            error.message ||
            "Unable to submit feedback.";

        status.className =
            "feedback-status error";

    }

}


window.openFeedbackModal =
    openFeedbackModal;

window.closeFeedbackModal =
    closeFeedbackModal;