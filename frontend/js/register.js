// =========================================
// AI STOCK MONITOR - REGISTER
// =========================================

document.addEventListener("DOMContentLoaded", () => {

    const registerForm = document.getElementById("register-form");

    const fullName = document.getElementById("full-name");
    const email = document.getElementById("email");
    const password = document.getElementById("password");
    const confirmPassword = document.getElementById("confirm-password");

    const strengthMeter = document.getElementById("strength-meter");

    if (!registerForm) return;

    // ----------------------------
    // Password Strength Meter
    // ----------------------------

    password.addEventListener("input", (e) => {

        const value = e.target.value;

        let strength = 0;

        if (value.length >= 8) strength++;

        if (/[A-Z]/.test(value)) strength++;

        if (/[a-z]/.test(value)) strength++;

        if (/[0-9]/.test(value)) strength++;

        if (/[^A-Za-z0-9]/.test(value)) strength++;

        strengthMeter.className = "strength-meter";

        if (strength <= 2) {

            strengthMeter.classList.add("weak");

        }

        else if (strength <= 4) {

            strengthMeter.classList.add("medium");

        }

        else {

            strengthMeter.classList.add("strong");

        }

    });

    // ----------------------------
    // Register
    // ----------------------------

    registerForm.addEventListener("submit", async (e) => {

        e.preventDefault();

        // Password Match

        if (password.value !== confirmPassword.value) {

            alert("Passwords do not match.");

            return;

        }

        const button = registerForm.querySelector("button");

        const originalText = button.innerText;

        button.innerText = "Creating Account...";

        button.disabled = true;

        try {

            const response = await fetch("http://localhost:5000/api/auth/register", {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    fullName: fullName.value.trim(),

                    email: email.value.trim(),

                    password: password.value

                })

            });

            const data = await response.json();

            if (response.ok) {

                alert(data.message);

                registerForm.reset();

                strengthMeter.className = "strength-meter";

                window.location.href = "login.html";

            }

            else {

                alert(data.message);

            }

        }

        catch (error) {

            console.error(error);

            alert("Unable to connect to backend.");

        }

        finally {

            button.innerText = originalText;

            button.disabled = false;

        }

    });

});