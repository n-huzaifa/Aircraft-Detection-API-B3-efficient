const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const resultContainer = document.getElementById('result');
const imagePreview = document.getElementById('imagePreview');
const imageElement = document.getElementById('imageElement');
const loadingSpinner = document.getElementById('loadingSpinner');

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            imageElement.src = reader.result;
            imageElement.style.display = 'block';
        };
        reader.readAsDataURL(file);
        uploadButton.disabled = false;
    }
});

uploadButton.addEventListener('click', async (event) => {
    event.preventDefault();  // Prevent page refresh

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    loadingSpinner.style.display = 'block';
    resultContainer.innerHTML = '';

    try {
        const response = await fetch('http://127.0.0.1:8000/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        console.log(response)
        if (data.error) {
            resultContainer.textContent = `Error: ${data.error}`;
        } else {
            const { predicted_class_name, predicted_probability, plane_detail } = data;
            resultContainer.innerHTML = `<div>
                <h3>Predicted Class: ${predicted_class_name}</h3>
                <p>Probability: ${(predicted_probability * 100).toFixed(2)}%</p>
                <p>Details: ${plane_detail.replace(/\n/g, '<br>')}</p>
            </div>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultContainer.textContent = 'An error occurred during upload.';
    } finally {
        loadingSpinner.style.display = 'none';
    }
});
