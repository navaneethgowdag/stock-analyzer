// =========================================
// AI STOCK MONITOR - CORE JS
// =========================================

// --- Theme Toggle Logic ---
function initTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (!themeToggleBtn) return;

    const htmlEl = document.documentElement;
    const icon = themeToggleBtn.querySelector('i');

    // Check for saved theme in localStorage
    const savedTheme = localStorage.getItem('theme');
    // Check for system preference
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Apply initial theme
    if (savedTheme) {
        htmlEl.setAttribute('data-theme', savedTheme);
    } else if (!systemPrefersDark) {
        htmlEl.setAttribute('data-theme', 'light');
    }

    // Update icon based on current theme
    function updateIcon() {
        const currentTheme = htmlEl.getAttribute('data-theme');
        if (currentTheme === 'dark') {
            icon.classList.remove('bi-sun-fill');
            icon.classList.add('bi-moon-stars-fill');
        } else {
            icon.classList.remove('bi-moon-stars-fill');
            icon.classList.add('bi-sun-fill');
        }
    }
    
    updateIcon();

    // Toggle theme on click
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlEl.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        htmlEl.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateIcon();
    });
}

// --- Navbar Scroll Logic ---
function initNavbar() {

    const userNav = document.getElementById("user-nav");
    const userName = document.getElementById("user-name");

    const user = JSON.parse(localStorage.getItem("user"));

    if (user) {

        userNav.style.display = "block";

        userName.innerText = `Hi, ${user.full_name}`;

    }
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// --- Mobile Menu Logic ---
function initMobileMenu() {
    const menuBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (!menuBtn || !navLinks) return;

    menuBtn.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        const icon = menuBtn.querySelector('i');
        if (navLinks.classList.contains('active')) {
            icon.classList.remove('bi-list');
            icon.classList.add('bi-x');
        } else {
            icon.classList.remove('bi-x');
            icon.classList.add('bi-list');
        }
    });
}

// --- Scroll Reveal Animation ---
function initReveal() {
    const reveals = document.querySelectorAll('.reveal');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, { threshold: 0.1 });

    reveals.forEach(reveal => {
        observer.observe(reveal);
    });
}

// --- Scroll Progress Bar ---
function initScrollProgress() {
    const bar = document.getElementById('scroll-progress-bar');
    if (!bar) return;

    function update() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        bar.style.width = pct + '%';
    }

    window.addEventListener('scroll', update, { passive: true });
    update();
}

// --- Scroll-Reactive Floaters (coins, $ signs, bitcoin, arrows) ---
function initFloaters() {
    const floaters = document.querySelectorAll('.floater');
    if (!floaters.length) return;

    let ticking = false;

    function updateFloaters() {
        const scrollY = window.scrollY;

        floaters.forEach((el) => {
            const speed = parseFloat(el.dataset.speed || '0.15');
            const drift = parseFloat(el.dataset.drift || '0');
            const rotate = parseFloat(el.dataset.rotate || '0');

            const offsetY = -(scrollY * speed);
            const offsetX = Math.sin(scrollY / 260) * drift;
            const rot = (scrollY / 18) * rotate;

            el.style.transform =
                `translate(${offsetX}px, ${offsetY}px) rotate(${rot}deg)`;
        });

        ticking = false;
    }

    function onScroll() {
        if (!ticking) {
            requestAnimationFrame(updateFloaters);
            ticking = true;
        }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    updateFloaters();
}

// --- Parallax Mouse Movement (Hero) ---
function initParallax() {
    const hero = document.querySelector('.hero');
    if (!hero) return;

    const blobs = hero.querySelectorAll('.blob');
    
    hero.addEventListener('mousemove', (e) => {
        const x = (window.innerWidth / 2 - e.pageX) / 50;
        const y = (window.innerHeight / 2 - e.pageY) / 50;

        blobs.forEach((blob, index) => {
            const speed = (index + 1) * 0.5;
            blob.style.transform = `translate(${x * speed}px, ${y * speed}px)`;
        });
    });
}

// ==============================
// Authentication Navbar
// ==============================

function initAuthNavbar() {

    const loginNav = document.getElementById("login-nav");
    const registerNav = document.getElementById("register-nav");
    const logoutNav = document.getElementById("logout-nav");
    const logoutBtn = document.getElementById("logout-btn");

    const token = localStorage.getItem("token");

    if (token) {

        if (loginNav) loginNav.style.display = "none";

        if (registerNav) registerNav.style.display = "none";

        if (logoutNav) logoutNav.style.display = "block";

    }

    else {

        if (loginNav) loginNav.style.display = "block";

        if (registerNav) registerNav.style.display = "block";

        if (logoutNav) logoutNav.style.display = "none";

    }

    if (logoutBtn) {

        logoutBtn.addEventListener("click", () => {

            localStorage.removeItem("token");

            localStorage.removeItem("user");

            window.location.href = "login.html";

        });

    }

}

document.addEventListener("DOMContentLoaded", () => {

    initTheme();

    initNavbar();

    initMobileMenu();

    initReveal();

    initParallax();

    initAuthNavbar();

    initScrollProgress();

    initFloaters();

});