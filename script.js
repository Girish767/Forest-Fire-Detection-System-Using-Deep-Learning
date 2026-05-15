// Auto-dismiss flash messages
document.addEventListener('DOMContentLoaded', () => {
    const alerts = document.querySelectorAll('.card');
    alerts.forEach(alert => {
        if (alert.style.borderColor === 'rgb(255, 75, 75)') { // Check if it's an error message
            setTimeout(() => {
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 500);
            }, 5000);
        }
    });
});
