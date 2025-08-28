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
  
// const video = document.getElementById("video");
// const canvas = document.getElementById("overlay");
// const ctx = canvas.getContext("2d");

// function resizeCanvas() {
//   canvas.width = video.clientWidth;
//   canvas.height = video.clientHeight;
//   canvas.style.top = video.offsetTop + "px";
//   canvas.style.left = video.offsetLeft + "px";
//   canvas.style.pointerEvents = "none";
// }

// video.addEventListener("loadedmetadata", resizeCanvas);
// window.addEventListener("resize", resizeCanvas);

// // Fetch precomputed landmarks
// fetch("/static/video_landmarks.json")
//   .then(res => res.json())
//   .then(landmarksData => {

//     // Only keep frames with landmarks
//     const validFrames = landmarksData.filter(f => f.landmarks && f.landmarks.length);

//     function drawLandmarks() {
//       resizeCanvas(); // ensure canvas always matches video size

//       if (video.paused || video.ended) {
//         requestAnimationFrame(drawLandmarks);
//         return;
//       }

//       const time = video.currentTime;

//       // Find the closest frame with landmarks
//       let frameData = null;
//       let minDiff = Infinity;

//       for (let i = 0; i < validFrames.length; i++) {
//         const diff = Math.abs(validFrames[i].timestamp - time);
//         if (diff < minDiff) {
//           minDiff = diff;
//           frameData = validFrames[i];
//         }
//       }

//       // Clear previous drawings
//       ctx.clearRect(0, 0, canvas.width, canvas.height);

//       // Draw landmarks
//       if (frameData) {
//         frameData.landmarks.forEach(lm => {
//           const x = lm.x * video.clientWidth;
//           const y = lm.y * video.clientHeight;
//           ctx.beginPath();
//           ctx.arc(x, y, 5, 0, 2 * Math.PI);
//           ctx.fillStyle = "lime";
//           ctx.fill();
//         });
//       }

//       requestAnimationFrame(drawLandmarks);
//     }

//     video.addEventListener("play", () => {
//       drawLandmarks();
//     });

//   })
//   .catch(err => console.error("Failed to load landmarks:", err));
const video = document.getElementById("video");
const videoCaption = document.getElementById("videoCaption");

console.log("Script loaded, video element:", video);

if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
  video.setAttribute("playsinline", "");
  videoCaption.textContent = "Mobile Landmark Drawing Not Supported.";
  console.log("Mobile detected → playsinline applied");
}

const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
console.log("Canvas + context ready:", canvas, ctx);

function updateCanvasSize() {
  console.log("Updating canvas size...");
  const rect = video.getBoundingClientRect();
  const videoAspect = video.videoWidth / video.videoHeight;
  const rectAspect = rect.width / rect.height;

  let contentWidth, contentHeight, offsetX, offsetY;

  if (rectAspect > videoAspect) {
    contentHeight = rect.height;
    contentWidth = videoAspect * contentHeight;
    offsetX = (rect.width - contentWidth) / 2;
    offsetY = 0;
  } else {
    contentWidth = rect.width;
    contentHeight = contentWidth / videoAspect;
    offsetX = 0;
    offsetY = (rect.height - contentHeight) / 2;
  }

  canvas.width = contentWidth;
  canvas.height = contentHeight;
  canvas.style.top = offsetY + "px";
  canvas.style.left = offsetX + "px";

  console.log("Canvas resized:", { contentWidth, contentHeight, offsetX, offsetY });

  return { contentWidth, contentHeight, offsetX, offsetY };
}

let start, end;

console.log("Fetching landmarks JSON...");
fetch("/static/video_landmarks.json")
  .then(res => {
    console.log("Fetch response:", res.status);
    return res.json();
  })
  .then(data => {
    const filename = video.currentSrc.split("/").pop(); // get just the filename
    const frames = data[filename];

    if (!frames || !frames.length) {
      console.warn("No landmarks found for:", filename);
      return;
    }

    console.log("Loaded landmarks for", filename, "→", frames.length, "frames");

    const validFrames = frames.filter(f => f.landmarks && f.landmarks.length);

    function drawLandmarks() {
      console.log("drawLandmarks tick → paused?", video.paused, "ended?", video.ended);

      const { contentWidth, contentHeight, offsetX, offsetY } = updateCanvasSize();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!video.paused && !video.ended) {
        const time = video.currentTime - start;
        console.log("Video time:", video.currentTime, "Offset time:", time);

        let frameData = null;
        let minDiff = Infinity;

        for (let i = 0; i < validFrames.length; i++) {
          const diff = Math.abs(validFrames[i].timestamp - time);
          if (diff < minDiff) {
            minDiff = diff;
            frameData = validFrames[i];
          }
        }

        if (frameData) {
          console.log("Closest frame:", frameData.timestamp, "diff:", minDiff);
          frameData.landmarks.forEach((lm, idx) => {
            const x = offsetX + lm.x * contentWidth;
            const y = offsetY + lm.y * contentHeight;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
            if (idx === 0) {
              console.log("First landmark drawn at:", { x, y });
            }
          });
        } else {
          console.warn("No frameData found for current time");
        }
      }

      requestAnimationFrame(drawLandmarks);
    }

    window.addEventListener("resize", () => {
      console.log("Window resized");
      updateCanvasSize();
    });

    function initVideo() {
  const mov = video.dataset.mov == "True" || false;
  console.log("HELLO", mov)
  if (mov) {
    start = parseFloat(video.dataset.start) || 0;
    end = parseFloat(video.dataset.end) || 0;
  } else {
    start = 0;
    end = video.duration; // safe if metadata already loaded
  }

  console.log("Init video → Start:", start, "End:", end, "Mov:", mov);

  video.currentTime = start;

  drawLandmarks();

  // attach drawLandmarks now
  video.addEventListener("play", () => {
    console.log("Video play event fired → starting draw loop");
    drawLandmarks();
  });

  video.play().then(() => {
    console.log("Video playback started at", video.currentTime);
  }).catch(err => {
    console.error("Video play error:", err);
  });
}

// If metadata already loaded, init immediately
if (video.readyState >= 1) { // HAVE_METADATA
  console.log("Metadata already available, init immediately");
  initVideo();
} else {
  video.addEventListener("loadedmetadata", () => {
    console.log("loadedmetadata fired");
    initVideo();
  });
}
  })
  .catch(err => console.error("Failed to load landmarks JSON:", err));

// loop video
video.addEventListener("timeupdate", () => {
  console.log("timeupdate event. currentTime:", video.currentTime, "end:", end);
  if (end > 0 && video.currentTime >= end - 0.01) {
    console.log("Looping video back to start");
    video.currentTime = start;
    video.play();
  }
});




});

