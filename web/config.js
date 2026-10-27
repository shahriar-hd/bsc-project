/* GitHub Pages config — point this at your Hugging Face model repo.
   After running `.upload/convert_and_upload.py --repo <user/model>`,
   set repo below (or leave the placeholder: the page still works by
   loading a local .onnx file chosen with the "Load .onnx" button). */
window.MTL_CONFIG = {
  hfRepo: "shahriarhd/mtl-face-auth",      // your Hugging Face model repo
  modelFile: "model_best.onnx",            // exported by convert_and_upload.py
  thresholds: { deepfake: 0.6, spoof: 0.6, temporal: 0.8 },
  frames: 8,
  size: 224,
};
