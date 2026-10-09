# Demonstration data card

## Intended data

One personally collected, fixed-camera RGB video of a coloured tabletop object being pushed across the supplied calibration board. The submitted take is a 13.27 s, 1080 × 1920, 30 fps phone recording. The data is used to recover the path geometry that becomes the robot policy prior.

## Stored representation

`Demonstration.save()` writes a compressed NumPy archive containing:

- monotonic timestamps;
- calibrated object coordinates in metres;
- a per-sample detector confidence in `[0, 1]`;
- calibrated workspace size;
- SHA-256 of the source video;
- collection and extraction metadata, including whether the source is synthetic.

No video frames, audio, face data, or absolute filesystem paths are stored in the archive.

## Processing

ArUco centres define a planar homography, recomputed per frame when all four markers are visible. A configurable HSV mask is morphologically opened/closed. Candidate contours are scored using measured metric size, circularity, calibrated board bounds, and temporal continuity. Missing samples are interpolated, coordinates are smoothed, stationary lead-in/out is cropped, and the sequence is resampled with a shape-preserving cubic interpolator.

The public input is `data/public/personal_demo.MOV`. Its SHA-256 is `e6a12a9907155972615d5e0e45dcf46ebf3d3053fc088b32db35ec9b5d34b52d`; the same digest is stored in the derived demonstration and metrics. The applicant explicitly chose to publish the original recording, including its audio and camera metadata.

## Limitations

- Colour segmentation still assumes a controlled scene; metric and temporal gates reduce but do not eliminate distractor risk.
- A single homography assumes a planar workspace and a fixed camera.
- Confidence measures detector quality, not semantic correctness.
- One demonstration is evidence for one task, not a general manipulation dataset.
- The public synthetic artifact exists only for software verification and is tagged `synthetic: true` throughout the pipeline.
