// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get the start button element
    const startButton = document.getElementById('startButton');
    
    // Add click event listener to the start button
    startButton.addEventListener('click', function() {
        // Show loading state
        startButton.classList.add('loading');
        startButton.innerHTML = '<span>Starting...</span><div class="button-animation"></div>';
        
        // Make a request to start the backend service
        fetch('/start-gesture-controller', {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to start the gesture controller');
            }
            return response.json();
        })
        .then(data => {
            console.log('Gesture controller started:', data);
            // Update UI to show it's running
            startButton.classList.remove('loading');
            startButton.classList.add('running');
            startButton.innerHTML = '<span>Controller Active</span><div class="button-animation"></div>';
            
            // Show a notification
            showNotification('Gesture Controller is now active! Move your hand in front of the camera.');
        })
        .catch(error => {
            console.error('Error:', error);
            // Reset button on error
            startButton.classList.remove('loading');
            startButton.innerHTML = '<span>Try Again</span><div class="button-animation"></div>';
            
            // Show error notification
            showNotification('Failed to start the controller. Please check if your camera is connected.', 'error');
        });
    });
    
    // Function to show notification
    function showNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = message;
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Show with animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 500);
        }, 5000);
    }
});