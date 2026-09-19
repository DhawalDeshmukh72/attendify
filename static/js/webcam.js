/**
 * Minimal webcam helper: starts the camera into a <video>, and captures
 * the current frame to a base64 JPEG data URL via a hidden <canvas>.
 */
function initWebcam(videoId, canvasId) {
  const video = document.getElementById(videoId);
  const canvas = document.getElementById(canvasId);

  navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false })
    .then((stream) => {
      video.srcObject = stream;
      video.play();
    })
    .catch((err) => {
      const box = document.getElementById("webcam-error");
      if (box) {
        box.classList.remove("d-none");
        box.textContent = "Couldn't access the camera: " + err.message;
      }
    });

  return {
    capture() {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth || 480;
      canvas.height = video.videoHeight || 360;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.9);
    },
  };
}
