# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `EngineConfig.preview_interval`: attach the source image to `FrameResult`
  every Nth frame rather than all of them. Drawing the skeleton overlay costs
  ~0.7 ms on the detection thread, which is immaterial against ~15 ms of
  inference but not on hardware where inference already fills the frame budget.
  Only the preview is thinned — activity state, events and stats still update
  every frame. Skipped frames carry `image=None`, so `render()` returns `None`
  and consumers need no new branch.

## [0.2.0] - 2026-09-06

### Added

- `left_thumbs_up` / `right_thumbs_up` activities: hand closed with the thumb
  extended and pointing up. Judged on the thumb alone, since a thumbs-up curls
  the same four fingers a fist does — it therefore suppresses `fist_closed`
  rather than firing alongside it.
- `HandFeatures.thumb_extension`, `HandFeatures.thumb_up` and
  `HandFeatures.outer_curl`, the geometry the above is built on.
- `Tuning.thumbs_up_extension_min`, `Tuning.thumbs_up_direction_min` and
  `Tuning.pinch_outer_curl_min`.

### Changed

- `HandFeatures` gained three fields, so constructing one **positionally** now
  breaks. It is handed to you by the engine rather than built by callers, so
  this should not reach normal use; construct it by keyword if you do build one.
- `fist_closed` no longer holds while a thumbs-up does, and `pinch` no longer
  holds while the hand is closed. If you had a binding relying on the old
  co-firing, it will now see only the more specific gesture.

### Fixed

- Pinch no longer fires on a closed fist. A fist folds the thumb across the
  curled index, putting the two tips as close together as a deliberate pinch
  does, so the thumb-to-index gap alone could not separate them. A pinch is now
  corroborated by the fingers it does not use — the middle, ring and pinky must
  not themselves be curled into a fist.
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

[0.2.0]: https://github.com/ikbal-nayem/motionsense-sdk/releases/tag/v0.2.0
[0.1.0]: https://github.com/ikbal-nayem/motionsense-sdk/releases/tag/v0.1.0
