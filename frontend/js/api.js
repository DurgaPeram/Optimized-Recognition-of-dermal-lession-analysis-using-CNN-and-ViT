const API_URL = "http://127.0.0.1:8000/predict";

document.getElementById("imageInput")?.addEventListener("change", function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const preview = document.getElementById("preview");
        preview.src = e.target.result;
        preview.style.display = "block";
    };

    if (file) reader.readAsDataURL(file);
});

async function analyzeImage() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(API_URL, {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    document.getElementById("prediction").innerText = data.prediction;
    document.getElementById("confidence").innerText = data.confidence.toFixed(2);

    const riskBox = document.getElementById("riskBox");
    const riskLevel = document.getElementById("riskLevel");

    if (data.prediction === "Melanoma") {
        riskBox.style.background = "red";
        riskLevel.innerText = "HIGH";
    } else if (data.prediction === "Basal Cell Carcinoma") {
        riskBox.style.background = "orange";
        riskLevel.innerText = "MODERATE";
    } else {
        riskBox.style.background = "green";
        riskLevel.innerText = "LOW";
    }

    document.getElementById("resultCard").style.display = "block";

    localStorage.setItem("lastPrediction", data.prediction);
}