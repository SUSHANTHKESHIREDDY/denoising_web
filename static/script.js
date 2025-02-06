document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('image-input');
    
    if (fileInput.files.length > 0) {
        formData.append('image', fileInput.files[0]);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            const imageUrl = result.output_image_url;
            const outputImage = document.getElementById('output-image');
            outputImage.src = imageUrl;
            outputImage.style.display = 'block';
        } else {
            alert('Image upload failed. Please try again.');
        }
    } else {
        alert('Please select an image to upload.');
    }
});
