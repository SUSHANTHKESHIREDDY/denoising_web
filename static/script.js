document.getElementById('image-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const previewImage = document.getElementById('preview-image');
        previewImage.src = URL.createObjectURL(file);
        previewImage.style.display = 'block';
    }
});

document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('image-input');
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('/denoise', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (result.output_image_url) {
            const outputImage = document.getElementById('output-image');
            outputImage.src = result.output_image_url;
            outputImage.style.display = 'block';

            // Show and set download button
            const downloadBtn = document.getElementById('download-btn');
            downloadBtn.href = result.output_image_url;
            downloadBtn.style.display = 'inline-block';
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
});
