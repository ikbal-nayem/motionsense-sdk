# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Pinch no longer fires on a thumb and index that are far apart in depth.
  Hand ratios (`pinch`, `curl`, hand scale) were measured on the image plane
  only, so a gap lying along the view direction all but vanished in projection
  and read as a pinch; they are now measured in 3D.

## [0.1.0] - 2026-09-05

Initial release.

### Added

- `MotionEngine`: capture, inference, feature extraction, recognition and event
  dispatch behind a listener API (`on`, `on_start`, `on_end`, `on_update`,
  `on_trigger`, `on_frame`, `on_error`).
- 25 built-in activities across arms, posture, jump/wave/swipe and hand poses,
  described in `motionsense.catalog`.
- Dual MediaPipe backend support, selected automatically at runtime: the
  legacy `mediapipe.solutions` API (0.10.x) and the Tasks API (1.0+).
- Body-frame normalisation (`motionsense.features`) so thresholds mean the
  same physical thing regardless of camera distance, aspect ratio or subject
  position.
- 1-Euro landmark filtering, windowed-slope velocity estimation, and
  Schmitt-trigger hysteresis gates with dwell-time debouncing
  (`motionsense.mathx`).
- Per-subject `Calibration`, recorded from a short neutral-standing sample.
- `CameraSource` with a latest-frame capture policy, cold-start priming and
  transient-dropout tolerance; `VideoFileSource`, `IterableSource` and
  `CallableSource` for offline and custom inputs.
- Optional keyboard/mouse output (`motionsense.bindings`, `pip install
  motionsense[keys]`) and a swappable `ActionRouter` for data-driven mappings.
- Optional OpenCV preview rendering (`motionsense.draw`).
- 140+ tests covering the math layer, feature normalisation, recognizers,
  engine lifecycle and both MediaPipe backends.

[0.1.0]: https://github.com/ikbal-nayem/motionsense-sdk/releases/tag/v0.1.0
