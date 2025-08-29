const videoInput = document.getElementById('videoInput');
const videoDemo = document.getElementById('videoDemo');
const videoPreview = document.getElementById('videoPreview');
const startSlider = document.getElementById('startSlider');
const endSlider = document.getElementById('endSlider');
const startTimeDisplay = document.getElementById('startTime');
const endTimeDisplay = document.getElementById('endTime');
const startInput = document.getElementById('startInput');
const endInput = document.getElementById('endInput');
const sliderSection = document.getElementById('sliderSection');
const timeLabels = document.getElementById('timeLabels');
const sliderTrack = document.getElementById('sliderTrack');
const videoWrapper = document.getElementById('videoWrapper');
const startLabel = document.getElementById('startLabel')
const endLabel = document.getElementById('endLabel')
const uploadButton = document.getElementById('uploadButton')
const trimMessage = document.getElementById('trimMessage')
const lite = document.getElementById('lite')
const heavy = document.getElementById('heavy')
const selectVersion = document.getElementById('selectVersion')
const same = document.getElementById('same')

document.addEventListener("DOMContentLoaded", () => {
  if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
      videoDemo.setAttribute("playsinline", "");
    }
});

function updateEstimate() {
  const estimateText = document.getElementById('estimatedTime');
  const modelType = document.getElementById('modelType').value;

  const start = parseFloat(startSlider.value);
  const end = parseFloat(endSlider.value);
  const duration = end - start;

  let time = 1;

  if (!isNaN(duration)) {
    if (modelType === 'lite') {
      time = Math.ceil(duration * 2);
    } else if (modelType === 'heavy') {
      time = Math.ceil(duration * 5);
    }

    estimateText.textContent = `Estimated time: ~ ${time} seconds`;
  }
}

videoInput.addEventListener('change', function () {
  const file = this.files[0];
  if (file) {
    const url = URL.createObjectURL(file);

    videoPreview.src = url;
    
    if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
      videoPreview.setAttribute("playsinline", "");
    }

    videoWrapper.hidden = false;
    startLabel.hidden = false;
    endLabel.hidden = false;
    startTimeDisplay.hidden = false;
    endTimeDisplay.hidden = false;
    trimMessage.hidden = false;
    lite.hidden = false;
    heavy.hidden = false;
    selectVersion.hidden = false;
    same.hidden = false;




    videoPreview.onloadedmetadata = function () {
      const duration = videoPreview.duration;

      // Set max for sliders
      startSlider.max = endSlider.max = duration;
      startSlider.value = 0;
      endSlider.value = duration;

      // Set min just in case (optional)
      startSlider.min = 0;
      endSlider.min = 0;

      startTimeDisplay.textContent = '0';
      endTimeDisplay.textContent = duration.toFixed(1);
      startInput.value = 0;
      endInput.value = duration.toFixed(1);

      updateSliderTrack();

      sliderSection.hidden = false;
      timeLabels.hidden = false;

      videoPreview.play();
    };
  }
});

function updateSliderTrack() {
  const duration = videoPreview.duration;
  const start = parseFloat(startSlider.value);
  const end = parseFloat(endSlider.value);

  // Calculate percentages
  const startPercent = (start / duration) * 100;
  const endPercent = (end / duration) * 100;

  // Set left and width of the colored track div
  sliderTrack.style.left = `${startPercent}%`;
  sliderTrack.style.width = `${endPercent - startPercent}%`;
}

startSlider.addEventListener('input', () => {
  let start = parseFloat(startSlider.value);
  const end = parseFloat(endSlider.value);

  if (start >= end) {
    start = end - 0.1;
    startSlider.value = start;
  }

  startTimeDisplay.textContent = start.toFixed(1);
  startInput.value = start;

  updateSliderTrack();
  videoPreview.currentTime = start;
  videoPreview.play()
});

let isHandlingEndInput = false;

endSlider.addEventListener('input', () => {
  if (isHandlingEndInput) return; 
  isHandlingEndInput = true;

  let newEnd = parseFloat(endSlider.value);
  let start = parseFloat(startSlider.value);
  let currentEnd = parseFloat(endInput.value);

  if (newEnd < start) {
    startSlider.value = newEnd;
    startTimeDisplay.textContent = newEnd.toFixed(1);
    startInput.value = newEnd;
    videoPreview.currentTime = newEnd - 0.5;

    endSlider.value = currentEnd;
    endTimeDisplay.textContent = currentEnd.toFixed(1);
    endInput.value = currentEnd;
  } else {
    endTimeDisplay.textContent = newEnd.toFixed(1);
    endInput.value = newEnd;
    videoPreview.currentTime = newEnd - 0.5;
  }

  updateSliderTrack();

  setTimeout(() => {
    isHandlingEndInput = false;
  }, 20);
});

videoPreview.addEventListener('timeupdate', () => {
  const start = parseFloat(startSlider.value);
  const end = parseFloat(endSlider.value);

  if (videoPreview.currentTime < start) {
    videoPreview.currentTime = start;
  }

  if (videoPreview.currentTime >= end) {
    videoPreview.currentTime = start;
  }
});

function selectModel(type) {
  const modelInput = document.getElementById('modelType');
  const estimateText = document.getElementById('estimatedTime');
  const buttons = document.querySelectorAll('.model-btn');
  uploadButton.hidden = false;
  estimateText.hidden = false;

  const start = parseFloat(startSlider.value);
  const end = parseFloat(endSlider.value);
  const duration = end - start

  modelInput.value = type;
  let time = 1

  buttons.forEach(btn => btn.classList.remove('active'));
  if (type === 'lite') {
    time = Math.ceil(duration * 1.5 + 4)
    estimateText.textContent = `Estimated time: ~ ${time} seconds`;
    document.querySelector('.model-btn:nth-child(1)').classList.add('active');
  } else {
    time = Math.ceil(duration * 2.2 + 4)
    estimateText.textContent = `Estimated time: ~ ${time} seconds`;
    document.querySelector('.model-btn:nth-child(2)').classList.add('active');
  }
}

const uploadForm = document.getElementById("uploadForm");

uploadForm.addEventListener("submit", function (e) {
  if (e.submitter && e.submitter.id === "uploadButton") {
    const overlay = document.getElementById("loadingOverlay");
    const message = overlay.querySelector("p");
    overlay.hidden = false;

    const type = document.getElementById('modelType').value
    const start = parseFloat(startSlider.value);
    const end = parseFloat(endSlider.value);
    const durationTime = end - start;
    let time = 1 * 1000

    if (type === 'lite') {
      time = Math.ceil(durationTime * 1.5 + 4) * 1000

    } else {
      time = Math.ceil(durationTime * 2.2 + 4) * 1000
    }

    const steps = [
      { text: "Downloading video...", duration: time/8},
      { text: "Trimming video...", duration: time/8 },
      { text: "Creating landmarks...", duration: time/4 },
      { text: "Making prediction...", duration: time/6 },
      { text: "Drawing landmarks...", duration: time/3 }
    ];

    let totalTime = 0;

    steps.forEach(step => {
      totalTime += step.duration;
      setTimeout(() => {
        message.textContent = step.text;
      }, totalTime - step.duration);
    });
  }
});

startSlider.addEventListener('input', updateEstimate);
endSlider.addEventListener('input', updateEstimate);