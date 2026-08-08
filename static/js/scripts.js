/* ============================================================
   Ringworm Detection System — Frontend logic
   ============================================================ */
(function () {
    "use strict";

    // ---------- Navigation toggle ----------
    const navToggle = document.getElementById("navToggle");
    const navMenu = document.getElementById("navMenu");

    function closeMobileMenu() {
        if (!navToggle) return;
        navToggle.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
        if (navMenu) navMenu.classList.remove("open");
    }

    if (navToggle && navMenu) {
        navToggle.addEventListener("click", function () {
            const open = navMenu.classList.toggle("open");
            navToggle.classList.toggle("open", open);
            navToggle.setAttribute("aria-expanded", String(open));
        });

        navMenu.addEventListener("click", function (event) {
            if (event.target.tagName === "A") {
                closeMobileMenu();
            }
        });
    }

    // ---------- State ----------
    let selectedFile = null;
    let selectedObjectURL = null;
    let isProcessing = false;

    const ALLOWED = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"];
    const MAX_SIZE = 5 * 1024 * 1024; // 5 MB

    // ---------- Detection page elements ----------
    const dropZone = document.getElementById("dropZone");
    const imageInput = document.getElementById("imageInput");
    const dropEmpty = document.getElementById("dropEmpty");
    const dropPreview = document.getElementById("dropPreview");
    const previewImage = document.getElementById("previewImage");
    const replaceBtn = document.getElementById("replaceBtn");
    const removeBtn = document.getElementById("removeBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const validationMessage = document.getElementById("validationMessage");
    const loadingPanel = document.getElementById("loadingPanel");
    const loadingSteps = document.getElementById("loadingSteps");
    const errorPanel = document.getElementById("errorPanel");
    const errorTitle = document.getElementById("errorTitle");
    const errorMessage = document.getElementById("errorMessage");
    const resultCard = document.getElementById("resultCard");
    const resultTime = document.getElementById("resultTime");
    const statusBadge = document.getElementById("statusBadge");
    const resultOriginal = document.getElementById("resultOriginal");
    const resultAnnotated = document.getElementById("resultAnnotated");
    const statStatus = document.getElementById("statStatus");
    const statConfidence = document.getElementById("statConfidence");
    const statCount = document.getElementById("statCount");
    const confidenceFill = document.getElementById("confidenceFill");
    const confidencePct = document.getElementById("confidencePct");
    const downloadBtn = document.getElementById("downloadBtn");
    const againBtn = document.getElementById("againBtn");

    // If this is not the detection page, nothing more to do.
    if (!dropZone || !imageInput) return;

    // ---------- Helpers ----------
    function setMessage(text) {
        if (validationMessage) validationMessage.textContent = text || "";
    }

    function hideMessage() {
        setMessage("");
    }

    function showError(title, message) {
        if (errorTitle) errorTitle.textContent = title || "Unable to analyze image";
        if (errorMessage) errorMessage.textContent = message || "Something went wrong while processing the image. Please try again.";
        if (errorPanel) errorPanel.hidden = false;
    }

    function hideError() {
        if (errorPanel) errorPanel.hidden = true;
    }

    function showLoading() {
        if (loadingPanel) loadingPanel.hidden = false;
        if (loadingSteps) loadingSteps.textContent = "Uploading";
    }

    function setLoadingStep(step) {
        if (loadingSteps) loadingSteps.textContent = step;
    }

    function hideLoading() {
        if (loadingPanel) loadingPanel.hidden = true;
    }

    function clearResult() {
        if (resultCard) resultCard.hidden = true;
    }

    function resetInterface() {
        if (selectedObjectURL) {
            URL.revokeObjectURL(selectedObjectURL);
            selectedObjectURL = null;
        }
        selectedFile = null;
        isProcessing = false;

        hideMessage();
        hideError();
        hideLoading();
        clearResult();

        if (imageInput) imageInput.value = "";
        if (dropPreview) dropPreview.hidden = true;
        if (dropEmpty) dropEmpty.hidden = false;
        if (analyzeBtn) analyzeBtn.disabled = true;
    }

    // ---------- File selection ----------
    function validateFile(file) {
        if (!file) {
            setMessage("No image selected. Please choose an image to analyze.");
            return false;
        }

        if (!ALLOWED.includes(file.type)) {
            setMessage("Unsupported file type. Please upload a JPG, JPEG, PNG, WEBP or GIF image.");
            return false;
        }

        if (file.size > MAX_SIZE) {
            setMessage("File is too large. Maximum allowed size is 5 MB.");
            return false;
        }

        return true;
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            if (previewImage) previewImage.src = event.target.result;
            if (previewImage) previewImage.alt = "Preview of " + file.name;
            if (dropEmpty) dropEmpty.hidden = true;
            if (dropPreview) dropPreview.hidden = false;
        };
        reader.readAsDataURL(file);
    }

    function handleFile(file) {
        hideError();
        if (validateFile(file)) {
            hideMessage();
            selectedFile = file;
            showPreview(file);
            if (analyzeBtn) analyzeBtn.disabled = false;
        } else {
            selectedFile = null;
            if (analyzeBtn) analyzeBtn.disabled = true;
            if (dropPreview) dropPreview.hidden = true;
            if (dropEmpty) dropEmpty.hidden = false;
            if (imageInput) imageInput.value = "";
        }
    }

    // ---------- Events: file input ----------
    imageInput.addEventListener("change", function (event) {
        const file = event.target.files && event.target.files[0];
        if (file) handleFile(file);
    });

    // ---------- Events: keyboard on drop zone ----------
    dropZone.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            imageInput.click();
        }
    });

    // ---------- Events: drag & drop ----------
    ["dragenter", "dragover"].forEach(function (name) {
        dropZone.addEventListener(name, function (event) {
            event.preventDefault();
            event.stopPropagation();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(function (name) {
        dropZone.addEventListener(name, function (event) {
            event.preventDefault();
            event.stopPropagation();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", function (event) {
        const files = event.dataTransfer && event.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Prevent default browser behavior when dropping outside the zone.
    ["dragover", "drop"].forEach(function (name) {
        document.addEventListener(name, function (event) {
            event.preventDefault();
        });
    });

    // ---------- Events: replace / remove ----------
    if (replaceBtn) {
        replaceBtn.addEventListener("click", function (event) {
            event.stopPropagation();
            imageInput.click();
        });
    }

    if (removeBtn) {
        removeBtn.addEventListener("click", function (event) {
            event.stopPropagation();
            resetInterface();
        });
    }

    // ---------- Render result ----------
    function renderResult(data) {
        const detected = data.detected === true;
        const confidence = Math.round((data.confidence || 0) * 100);
        const count = data.detection_count || (data.boxes ? data.boxes.length : 0);

        if (statusBadge) {
            statusBadge.textContent = detected ? "Ringworm Detected" : "No Ringworm Detected";
            statusBadge.className = "status-badge " + (detected ? "detected" : "clean");
        }

        if (statStatus) statStatus.textContent = detected ? "Ringworm Detected" : "No Ringworm Detected";
        if (statConfidence) statConfidence.textContent = confidence + "%";
        if (statCount) statCount.textContent = String(count);

        if (confidenceFill) confidenceFill.style.width = confidence + "%";
        if (confidencePct) confidencePct.textContent = confidence + "%";

        if (resultTime) {
            resultTime.textContent = data.time_seconds ? "Processed in " + data.time_seconds + " seconds" : "";
        }

        if (resultAnnotated) resultAnnotated.src = data.result_image_url || "";

        if (resultOriginal) {
            if (selectedFile) {
                if (selectedObjectURL) URL.revokeObjectURL(selectedObjectURL);
                selectedObjectURL = URL.createObjectURL(selectedFile);
                resultOriginal.src = selectedObjectURL;
            } else {
                resultOriginal.src = "";
            }
        }

        if (downloadBtn) {
            downloadBtn.href = data.result_image_url || "#";
        }

        if (resultCard) resultCard.hidden = false;
        if (resultCard) resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ---------- Analyze ----------
    async function analyze() {
        if (isProcessing) return;
        if (!selectedFile) {
            setMessage("Please select an image before analyzing.");
            if (analyzeBtn) analyzeBtn.disabled = true;
            return;
        }

        hideError();
        hideMessage();
        clearResult();
        isProcessing = true;
        if (analyzeBtn) analyzeBtn.disabled = true;
        showLoading();

        const formData = new FormData();
        formData.append("image", selectedFile);
        const url = dropZone.dataset.predictUrl || "/api/predict";

        try {
            setLoadingStep("Uploading");
            await new Promise(function (resolve) {
                setTimeout(resolve, 250);
            });

            setLoadingStep("Processing");
            await new Promise(function (resolve) {
                setTimeout(resolve, 250);
            });

            const controller = new AbortController();
            const timeoutId = setTimeout(function () {
                controller.abort();
            }, 60000);

            const response = await fetch(url, {
                method: "POST",
                body: formData,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            setLoadingStep("Detecting");
            await new Promise(function (resolve) {
                setTimeout(resolve, 250);
            });

            if (!response.ok) {
                let data = null;
                try {
                    data = await response.json();
                } catch (e) {
                    data = null;
                }
                const message = (data && data.error) || "Unable to analyze image. Something went wrong while processing the image. Please try again.";
                throw new Error(message);
            }

            const result = await response.json();

            setLoadingStep("Preparing Results");
            await new Promise(function (resolve) {
                setTimeout(resolve, 250);
            });

            renderResult(result);
        } catch (err) {
            const message =
                err && err.name === "AbortError"
                    ? "The analysis took too long. Please try again."
                    : err && err.message
                    ? err.message
                    : "Unable to analyze image. Something went wrong while processing the image. Please try again.";
            showError("Unable to analyze image", message);
        } finally {
            hideLoading();
            isProcessing = false;
            if (analyzeBtn) analyzeBtn.disabled = !selectedFile;
        }
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", analyze);
    }

    // ---------- Analyze another image ----------
    if (againBtn) {
        againBtn.addEventListener("click", function () {
            resetInterface();
            if (dropZone) dropZone.scrollIntoView({ behavior: "smooth", block: "center" });
        });
    }

    // ---------- Init ----------
    resetInterface();
})();
