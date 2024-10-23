document.getElementById('file-upload').addEventListener('change', handleFileChange);
document.getElementById('upload-form').addEventListener('submit', handleSubmit);
document.getElementById('clear-btn').addEventListener('click', clearImage);
document.getElementById('close-popup').addEventListener('click', closePopup);

let selectedFile = null;

function handleFileChange(event) {
  const file = event.target.files[0];
  if (file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
      const previewImg = document.getElementById('preview');
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
      
      document.getElementById('clear-btn').classList.remove('hidden');
      document.getElementById('submit-btn').disabled = false;
    };
    reader.readAsDataURL(file);
  }
}

function handleSubmit(event) {
  event.preventDefault();
  
  if (!selectedFile) return;
  
  const formData = new FormData();
  formData.append('file', selectedFile);
  
  document.getElementById('submit-btn').disabled = true;
  
  fetch('/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => response.text())
  .then(result => {
    // Show the popup with prediction result
    showPopup(result);
  })
  .catch(error => {
    console.error('Error:', error);
    alert('Something went wrong!');
  })
  .finally(() => {
    document.getElementById('submit-btn').disabled = false;
  });
}

function showPopup(result) {
  const modal = document.getElementById('popup-modal');
  const predictionText = document.getElementById('popup-prediction-result');
  
  predictionText.textContent = result;
  modal.classList.add('show');
}

function closePopup() {
  const modal = document.getElementById('popup-modal');
  modal.classList.remove('show');
  
  // Clear the form and reset UI
  clearImage();
}

function clearImage() {
  selectedFile = null;
  document.getElementById('preview').style.display = 'none';
  document.getElementById('clear-btn').classList.add('hidden');
  document.getElementById('submit-btn').disabled = true;
}
