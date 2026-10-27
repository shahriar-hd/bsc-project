/* MTL ONNX in-browser inference. No build step, no bundler.
   Loads model from HF repo (config.js) or a user-picked local .onnx file,
   samples 8 frames from a video/camera capture, center-crops + ImageNet-
   normalizes them, runs ONNX Runtime Web, returns 3 sigmoid scores. */
(function (global) {
  'use strict';
  const cfg = () => global.MTL_CONFIG || {};
  const state = { session: null, modelSource: '', loading: null, faceDetector: null };
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  function hfUrl() {
    const c = cfg();
    return `https://huggingface.co/${c.hfRepo}/resolve/main/${c.modelFile || 'model_best.onnx'}`;
  }
  async function ensureOrt() {
    if (global.ort) return global.ort;
    await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js');
    if (!global.ort) throw new Error('onnxruntime-web failed to load');
    return global.ort;
  }
  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = () => rej(new Error('script load failed: ' + src));
      document.head.appendChild(s);
    });
  }
  async function loadModelFromUrl(url, label) {
    const ort = await ensureOrt();
    if (state.session && state.modelSource === url) return state.session;
    if (state.loading) await state.loading;
    state.loading = (async () => {
      const sess = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
      state.session = sess; state.modelSource = url;
      return sess;
    })();
    try { return await state.loading; }
    finally { state.loading = null; }
  }
  async function loadDefaultModel() { return loadModelFromUrl(hfUrl(), 'huggingface'); }
  async function loadLocalModel(file) {
    const url = URL.createObjectURL(file);
    return loadModelFromUrl(url, file.name);
  }
  function modelStatus() {
    if (state.session) return { loaded: true, source: state.modelSource };
    return { loaded: false, source: '' };
  }
  // ---- frame sampling: even spread + face-box stability gate ----
  async function sampleFrames(videoEl, n, size) {
    n = n || cfg().frames || 8; size = size || cfg().size || 224;
    const dur = isFinite(videoEl.duration) ? videoEl.duration : 0;
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const thumbs = [];
    const boxes = [];
    const box = await detectFaceBox(videoEl);
    for (let i = 0; i < n; i++) {
      const t = dur > 0 ? (dur * (i + 0.5)) / n : 0;
      await seekTo(videoEl, t);
      const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
      let sx = 0, sy = 0, sw = vw, sh = vh;
      if (box) {
        const pad = 0.15;
        const bw = box.w * (1 + pad * 2), bh = box.h * (1 + pad * 2);
        const side = Math.max(bw, bh);
        const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
        sw = sh = Math.min(side, vw, vh);
        sx = Math.min(Math.max(cx - sw / 2, 0), vw - sw);
        sy = Math.min(Math.max(cy - sh / 2, 0), vh - sh);
      } else {
        const side = Math.min(vw, vh);
        sx = (vw - side) / 2; sy = (vh - side) / 2; sw = sh = side;
      }
      ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, size, size);
      thumbs.push(canvas.toDataURL('image/jpeg', 0.7));
      boxes.push(box);
    }
    return { thumbs: thumbs, faceFound: !!box };
  }
  function seekTo(videoEl, t) {
    return new Promise((res) => {
      if (!isFinite(videoEl.duration) || videoEl.duration === 0) return res();
      const done = () => { videoEl.removeEventListener('seeked', done); res(); };
      videoEl.addEventListener('seeked', done);
      try { videoEl.currentTime = Math.min(Math.max(t, 0), videoEl.duration - 0.05); }
      catch (e) { done(); }
      setTimeout(done, 1500);
    });
  }
  async function detectFaceBox(videoEl) {
    try {
      if ('FaceDetector' in global) {
        if (!state.faceDetector) { try { state.faceDetector = new global.FaceDetector(); } catch (e) { return null; } }
        const bmp = await createImageBitmap(videoEl);
        const faces = await state.faceDetector.detect(bmp);
        bmp.close && bmp.close();
        if (faces && faces.length) {
          const f = faces.sort((a, b) => (b.boundingBox.width * b.boundingBox.height) - (a.boundingBox.width * a.boundingBox.height))[0].boundingBox;
          const sx = videoEl.videoWidth / videoEl.clientWidth, sy = videoEl.videoHeight / videoEl.clientHeight;
          return { x: f.x * sx, y: f.y * sy, w: f.width * sx, h: f.height * sy };
        }
      }
    } catch (e) { /* fall through to center crop */ }
    return null;
  }
  // ---- draw sampled thumbs offscreen, build (1,8,3,224,224) tensor ----
  async function videoToTensor(videoEl, n, size) {
    n = n || cfg().frames || 8; size = size || cfg().size || 224;
    const frames = await sampleFrames(videoEl, n, size);
    const data = new Float32Array(1 * n * 3 * size * size);
    const c = document.createElement('canvas'); c.width = size; c.height = size;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    const tmp = document.createElement('video');
    for (let i = 0; i < n; i++) {
      await new Promise((res, rej) => {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0);
          const px = ctx.getImageData(0, 0, size, size).data;
          for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
            const p = (y * size + x) * 4;
            for (let ch = 0; ch < 3; ch++) {
              const v = px[p + ch] / 255;
              data[(i * 3 + ch) * size * size + y * size + x] = (v - MEAN[ch]) / STD[ch];
            }
          }
          res();
        };
        img.onerror = rej; img.src = frames.thumbs[i];
      });
    }
    return { data: data, thumbs: frames.thumbs, faceFound: frames.faceFound, n: n, size: size };
  }
  async function runTensor(t, onProgress, thrOverride) {
    const ort = await ensureOrt();
    const c = cfg();
    const sess = state.session || await loadDefaultModel();
    onProgress && onProgress('warming up model...');
    const input = new ort.Tensor('float32', t.data, [1, t.n, 3, t.size, t.size]);
    const feed = {}; feed[sess.inputNames[0]] = input;
    onProgress && onProgress('scoring 8 frames...');
    const out = await sess.run(feed);
    const get = (name) => { const k = Object.keys(out).find((x) => x.toLowerCase().includes(name)) || sess.outputNames[0]; return out[k].data[0]; };
    const scores = { deepfake: sigmoid(get('deepfake')), spoof: sigmoid(get('spoof')), temporal: sigmoid(get('temp')) };
    const th = thrOverride || global.MTL_CONFIG_THRESH || (c.thresholds || { deepfake: 0.6, spoof: 0.6, temporal: 0.8 });
    const flags = { deepfake: scores.deepfake >= th.deepfake, spoof: scores.spoof >= th.spoof, temporal: scores.temporal >= th.temporal };
    return { scores: scores, flags: flags, blocked: !!(flags.deepfake || flags.spoof || flags.temporal), faceFound: t.faceFound, thumbs: t.thumbs };
  }
  async function scoreVideoElement(videoEl, onProgress) {
    const t = await videoToTensor(videoEl, cfg().frames || 8, cfg().size || 224);
    return runTensor(t, onProgress);
  }
  global.MTL_ONNX = { loadDefaultModel, loadLocalModel, loadModelFromUrl, modelStatus, sampleFrames, videoToTensor, runTensor, scoreVideoElement, hfUrl };
})(window);
