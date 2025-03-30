document.addEventListener('DOMContentLoaded', function() {
    // File input validation and display
    const audioFileInput = document.getElementById('audio_file');
    const lectureFileInput = document.getElementById('lecture_file');
    const audioFileNameDisplay = document.getElementById('audio-file-name');
    const lectureFileNameDisplay = document.getElementById('lecture-file-name');
    const submitButton = document.getElementById('submit-button');
    const audioLoadingIndicator = document.getElementById('audio-loading-indicator');
    const lectureLoadingIndicator = document.getElementById('lecture-loading-indicator');
    
    // Processing state elements
    const processingOverlay = document.getElementById('processing-overlay');
    const uploadForm = document.getElementById('upload-form');
    
    // Download buttons on results page
    const downloadButtons = document.querySelectorAll('.download-btn');
    
    // Update file name displays when files are selected
    if (audioFileInput) {
        audioFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                const fileSize = (this.files[0].size / (1024 * 1024)).toFixed(2); // Size in MB
                audioFileNameDisplay.textContent = `${fileName} (${fileSize} MB)`;
                audioFileNameDisplay.classList.remove('text-muted');
                audioFileNameDisplay.classList.add('text-success');
                audioLoadingIndicator.classList.add('d-none');
                
                // Enable submit button if a file is selected
                submitButton.disabled = false;
            } else {
                resetAudioFileDisplay();
                
                // Disable submit button if no audio file is selected
                submitButton.disabled = true;
            }
        });
    }
    
    if (lectureFileInput) {
        lectureFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                const fileSize = (this.files[0].size / (1024 * 1024)).toFixed(2); // Size in MB
                lectureFileNameDisplay.textContent = `${fileName} (${fileSize} MB)`;
                lectureFileNameDisplay.classList.remove('text-muted');
                lectureFileNameDisplay.classList.add('text-success');
                lectureLoadingIndicator.classList.add('d-none');
            } else {
                resetLectureFileDisplay();
            }
        });
    }
    
    // Reset file displays
    function resetAudioFileDisplay() {
        audioFileNameDisplay.textContent = 'No file selected';
        audioFileNameDisplay.classList.remove('text-success');
        audioFileNameDisplay.classList.add('text-muted');
        audioLoadingIndicator.classList.add('d-none');
    }
    
    function resetLectureFileDisplay() {
        lectureFileNameDisplay.textContent = 'No file selected (optional)';
        lectureFileNameDisplay.classList.remove('text-success');
        lectureFileNameDisplay.classList.add('text-muted');
        lectureLoadingIndicator.classList.add('d-none');
    }
    
    // Show processing overlay when form is submitted
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Validate audio file
            if (!audioFileInput.files.length) {
                e.preventDefault();
                showAlert('Please select an audio file to transcribe', 'danger');
                return;
            }
            
            // Show processing overlay
            processingOverlay.classList.remove('d-none');
            
            // Add visual indicators that files are being processed
            audioLoadingIndicator.classList.remove('d-none');
            if (lectureFileInput.files.length > 0) {
                lectureLoadingIndicator.classList.remove('d-none');
            }
            
            // Disable submit button
            submitButton.disabled = true;
        });
    }
    
    // Download functionality on results page
    if (downloadButtons) {
        downloadButtons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                
                const contentType = this.dataset.contentType;
                const url = `/download/${contentType}`;
                
                // Show loading state
                this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Downloading...';
                this.disabled = true;
                
                fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        // Create a blob and download link
                        const blob = new Blob([data.content], { type: 'text/plain' });
                        const downloadUrl = URL.createObjectURL(blob);
                        
                        const a = document.createElement('a');
                        a.href = downloadUrl;
                        a.download = data.filename;
                        document.body.appendChild(a);
                        a.click();
                        
                        // Clean up
                        document.body.removeChild(a);
                        URL.revokeObjectURL(downloadUrl);
                        
                        // Reset button state
                        this.innerHTML = this.dataset.originalText || 'Download';
                        this.disabled = false;
                    })
                    .catch(error => {
                        console.error('Download error:', error);
                        showAlert('Error downloading file. Please try again.', 'danger');
                        
                        // Reset button state
                        this.innerHTML = this.dataset.originalText || 'Download';
                        this.disabled = false;
                    });
            });
            
            // Store original button text
            button.dataset.originalText = button.innerHTML;
        });
    }
    
    // Helper function to show alerts
    function showAlert(message, type) {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.role = 'alert';
        
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertsContainer.appendChild(alert);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => {
                alertsContainer.removeChild(alert);
            }, 150);
        }, 5000);
    }
    
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
});
