# HandTracking

2021 experiments with hand and body pose estimation on an NVIDIA Jetson-class GPU using TensorRT, plus socket-based video streaming and mouse-cursor control from hand gestures. A scratch collection of scripts, not a packaged project, not maintained.

It is built on NVIDIA's [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) samples. It does not use MediaPipe or contour-based landmark extraction.

## What's inside

| Files | Purpose |
|---|---|
| `mainHand.py`, `cursor_control.py` | Hand pose with trt_pose (`resnet18_baseline_att`, TensorRT-optimized via torch2trt); keypoints are fed to an sklearn SVM (`preprocessdata.py`) to classify gestures, and `pyautogui` moves/clicks the cursor |
| `mainBody.py`, `live_demo.py`, `live_demoV2.py` | Body pose demos adapted from the trt_pose live demo |
| `hand_pose.json`, `human_pose.json`, `gesture.json` | Keypoint topology and gesture class lists |
| `preprocess_coco_person.py`, `preprocessdata.py` | Dataset and feature preprocessing helpers |
| `imagiz*.py`, `socketIO*.py`, `tmpClient*.py`, `tmpServer*.py`, `socket_tests/`, `camera_client_test/`, `webcam*` | Experiments with streaming camera frames between a Jetson and a PC (imagiz, sockets, fake webcam) |
| `livePix2Pix.py`, `client_preprocess_pix2pix.py`, `pix2pix*.py`, `resize_model.py`, `TF-TRT.py`, `TRT.py`, `test_tensorflow/` | pix2pix and TensorFlow-TensorRT conversion tests |
| `pyQT/` | Small PyQt GUI prototype |
| `tests/` | Threading and streaming test scripts |

## Run

Needs a Jetson (or other CUDA machine) with TensorRT, PyTorch, `torch2trt` and `trt_pose` installed, and the pretrained `hand_pose_resnet18_att_244_244.pth` weights from trt_pose. The weights are not in this repo. Python packages are pinned in `requirements.txt` (TensorFlow 2.4.1 era).

```bash
python3 mainHand.py
```

I did not run this (no Jetson hardware here). Paths are relative to the working directory, and `mainHand.py` reads `preprocess/gesture.json` while the file sits in the repo root, so expect to adjust paths.

## Status

2021 experiment, abandoned. No results, benchmarks or tests are committed. Many `tmp*` scripts are earlier versions of each other.

## Credits

- NVIDIA-AI-IOT [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) (hand/body pose models and demo notebooks) and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt).
- NVIDIA TensorRT / TF-TRT sample code.
- The `imagiz` Python package for client/server frame streaming, and `pyautogui` for cursor control.
