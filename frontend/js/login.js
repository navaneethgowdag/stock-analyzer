// =========================================
// AI STOCK MONITOR - LOGIN
// =========================================

document.addEventListener("DOMContentLoaded", () => {

    const loginForm = document.getElementById("login-form");

    if (!loginForm) {
        console.error("Login form not found!");
        return;
    }

    loginForm.addEventListener("submit", async (e) => {

        e.preventDefault();

        // Get Elements
        const emailElement = document.getElementById("login-email");
        const passwordElement = document.getElementById("login-password");

        if (!emailElement || !passwordElement) {

            alert("Email or Password input not found.");

            console.error("Email Element:", emailElement);
            console.error("Password Element:", passwordElement);

            return;

        }

        // Get Values
        const email = emailElement.value.trim();
        const password = passwordElement.value;

        // Button
        const btn = loginForm.querySelector('button[type="submit"]');

        const originalText = btn.innerText;

        btn.innerText = "Logging In...";
        btn.disabled = true;

        try {

            const response = await fetch("http://localhost:5000/api/auth/login", {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    email,
                    password

                })

            });

            const data = await response.json();

            if (response.ok) {

                alert("Login Successful");

                console.log(data);

                // Save JWT
                localStorage.setItem("token", data.token);

                localStorage.setItem("user", JSON.stringify(data.user));

                // Redirect
                window.location.href = "index.html";

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

            btn.innerText = originalText;
            btn.disabled = false;

        }

    });

});