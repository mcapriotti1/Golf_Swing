document.addEventListener('DOMContentLoaded', function () {
  const predictionEl = document.getElementById('predictionText');
  const confidenceEl = document.getElementById('confidenceText');

  // Prediction flicker logic
  if (predictionEl) {
    const circle = document.querySelector('.progress-ring__circle');
    const finalPrediction = predictionEl.dataset.prediction; // "Pro" or "Amateur"
    let isPro = finalPrediction === 'Pro';
    let flickerCount = 0;

    const flickerInterval = setInterval(() => {
      // Toggle prediction value
      isPro = !isPro;
      predictionEl.textContent = isPro ? 'Pro' : 'Amateur';

      // Toggle class
      predictionEl.classList.remove('prediction-pro', 'prediction-amateur');
      predictionEl.classList.add(isPro ? 'prediction-pro' : 'prediction-amateur');

      circle.classList.remove('pro', 'amateur');
      circle.classList.add(isPro ? 'pro' : 'amateur');

      flickerCount += 1;
      if (flickerCount >= 16) {
        clearInterval(flickerInterval);
        // Set final value back
        predictionEl.textContent = finalPrediction;
        predictionEl.classList.remove('prediction-pro', 'prediction-amateur');
        predictionEl.classList.add(finalPrediction === 'Pro' ? 'prediction-pro' : 'prediction-amateur');
      }
    }, 150); // every 200ms


    if (confidenceEl) {
      const finalPrediction = predictionEl.dataset.prediction;
      const text = document.getElementById('confidenceCircleText');

      circle.classList.remove('pro', 'amateur');

      if (finalPrediction === 'Pro') {
        circle.classList.add('pro');
      } else if (finalPrediction === 'Amateur') {
        circle.classList.add('amateur');
      }

      const radius = 68;
      const circumference = 2 * Math.PI * radius;

      circle.style.strokeDasharray = `${circumference}`;
      circle.style.strokeDashoffset = `${circumference}`;

      let current = 0;

      function setProgress(percent) {
        const offset = circumference - (percent / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        text.textContent = `${percent}%`;
      }

      function animateProgress(finalPercent, speed=20) {
        if (current >= parseFloat(finalPercent)) return;
        current += 1;
        setProgress(current);
        setTimeout(() => animateProgress(finalPercent, speed), speed); // tweak speed
      }

      animateProgress((confidenceEl.dataset.confidence));
    }

  }
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  canvas.style.top = video.offsetTop + "px";
  canvas.style.left = video.offsetLeft + "px";
  canvas.style.pointerEvents = "none";
}

video.addEventListener("loadedmetadata", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

// Fetch precomputed landmarks
fetch("/static/video_landmarks.json")
  .then(res => res.json())
  .then(landmarksData => {

    // Only keep frames with landmarks
    const validFrames = landmarksData.filter(f => f.landmarks && f.landmarks.length);

    function drawLandmarks() {
      resizeCanvas(); // ensure canvas always matches video size

      if (video.paused || video.ended) {
        requestAnimationFrame(drawLandmarks);
        return;
      }

      const time = video.currentTime;

      // Find the closest frame with landmarks
      let frameData = null;
      let minDiff = Infinity;

      for (let i = 0; i < validFrames.length; i++) {
        const diff = Math.abs(validFrames[i].timestamp - time);
        if (diff < minDiff) {
          minDiff = diff;
          frameData = validFrames[i];
        }
      }

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw landmarks
      if (frameData) {
        frameData.landmarks.forEach(lm => {
          const x = lm.x * video.clientWidth;
          const y = lm.y * video.clientHeight;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "lime";
          ctx.fill();
        });
      }

      requestAnimationFrame(drawLandmarks);
    }

    video.addEventListener("play", () => {
      drawLandmarks();
    });

  })
  .catch(err => console.error("Failed to load landmarks:", err));

});

