const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const dropZone = document.getElementById("dropZone");
const uploadForm = document.getElementById("uploadForm");
const loadingSpinner = document.getElementById("loadingSpinner");

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function (event) {
        previewImage.src = event.target.result;
        previewImage.hidden = false;
    };
    reader.readAsDataURL(file);
}

imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        showPreview(file);
    }
});

uploadForm.addEventListener("submit", () => {
    loadingSpinner.hidden = false;
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
    });
});

dropZone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        showPreview(files[0]);
    }
});
