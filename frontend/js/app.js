window.onload = function() {

    const lastPrediction = localStorage.getItem("lastPrediction");

    if (!lastPrediction) return;

    const data = diseasesData[lastPrediction];

    if (!data) return;

    document.getElementById("diseaseName").innerText = lastPrediction;
    document.getElementById("risk").innerText = data.risk;
    document.getElementById("causes").innerText = data.causes;
    document.getElementById("symptoms").innerText = data.symptoms;
    document.getElementById("recovery").innerText = data.recovery;
    document.getElementById("treatment").innerText = data.treatment;
    document.getElementById("prevention").innerText = data.prevention;
    document.getElementById("immediate").innerText = data.immediate;
};