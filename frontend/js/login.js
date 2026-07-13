// =========================================
// AI STOCK MONITOR - LOGIN VALIDATION
// =========================================

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    
    if (!loginForm) return;

    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Simple visual feedback for frontend MVP
        const btn = loginForm.querySelector('button[type="submit"]');
        const originalText = btn.innerText;
        
        btn.innerText = 'Connecting...';
        btn.disabled = true;
        btn.style.opacity = '0.7';

        // Simulate API call placeholder
        setTimeout(() => {
            btn.innerText = 'Backend Not Connected';
            btn.style.background = 'var(--warning)';
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.style.background = '';
            }, 2000);
        }, 1000);
    });
});