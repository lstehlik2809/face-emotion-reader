# Face Reader

Face Reader uses your camera to estimate which of eight facial expression patterns your face most resembles. It shows live scores and can record a score timeline that you can export as a CSV. The scores describe visible expressions. They cannot reliably tell you how someone feels.

[Open Face Reader](https://lstehlik2809.github.io/face-emotion-reader/)

## Camera and data

Allow camera access after selecting **Start camera**. Models and browser libraries download when you start the camera for the first time. Video frames are processed in your browser. Recordings contain scores and timestamps.

## Teach Face Reader

With the camera running, open **Personalize scores** and select **Teach Neutral**. Sit and look at the screen as you normally do. If you use a low webcam or have a full beard, capture your examples with your usual setup and appearance. After the countdown, keep your brow, eyes, and mouth relaxed and hold still for about three seconds.

That capture saves both a Neutral example and a reference for how your relaxed face looks at that camera angle. Teach a second expression to help Face Reader distinguish between your examples. If you change cameras or move to a different usual position, retake Neutral. Examples are stored as numeric features in your browser. **Clear personalization** removes the examples and the camera-angle reference.

## How it works

[MediaPipe Face Landmarker](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/web/vision) tracks one face and provides movement features for teaching. The [HSEmotion EfficientNet-B0 classifier](https://github.com/sb-ai-lab/EmotiEffLib/tree/af833487321c3efdcb1768a91a6c656a1986fdf6/models/affectnet_emotions/onnx) scores the face crop using [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). The bundled checkpoint is `enet_b0_8_best_afew.onnx`, a compact model the authors evaluated on annotated face crops from photos and videos. Its benchmark results do not establish how well it performs on your webcam or with subtle expressions.

## Run locally

Serve this directory with a static server, for example `python -m http.server 8000`, and open [localhost:8000](http://localhost:8000). Use HTTPS when hosting the app anywhere other than localhost so the browser can grant camera access.
