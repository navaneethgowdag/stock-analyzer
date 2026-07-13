// =========================================
// AI STOCK MONITOR - REGISTER VALIDATION
// =========================================

document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('register-form');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const strengthMeter = document.getElementById('strength-meter');

    if (!registerForm) return;

    // Password Strength Meter
    passwordInput.addEventListener('input', (e) => {
        const val = e.target.value;
        let strength = 0;

        if (val.length >= 8) strength += 1;
        if (val.match(/[A-Z]/)) strength += 1;
        if (val.match(/[0-9]/)) strength += 1;
        if (val.match(/[^A-Za-z0-9]/)) strength += 1;

        // Reset classes
        strengthMeter.className = 'strength-meter';
        
        if (strength <= 1) {
            strengthMeter.classList.add('weak');
        } else if (strength <= 3) {
            strengthMeter.classList.add('medium');
        } else if (strength === 4) {
            strengthMeter.classList.add('strong');
        }
    });

    // Form Submit Validation
    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const password = passwordInput.value;
        const confirmPassword = confirmPasswordInput.value;

        if (password !== confirmPassword) {
            confirmPasswordInput.style.borderColor = 'var(--danger)';
            confirmPasswordInput.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.2)';
            
            // Reset after 2 seconds
            setTimeout(() => {
                confirmPasswordInput.style.borderColor = '';
                confirmPasswordInput.style.boxShadow = '';
            }, 2000);
            return;
        }

        // Simulate API call placeholder
        const btn = registerForm.querySelector('button[type="submit"]');
        const originalText = btn.innerText;
        
        btn.innerText = 'Creating Account...';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerText = 'Backend Not Connected';
            btn.style.background = 'var(--warning)';
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.disabled = false;
                btn.style.background = '';
            }, 2000);
        }, 1000);
    });
});