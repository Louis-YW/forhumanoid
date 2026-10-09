# Demonstration data card

## Intended data

One personally collected, fixed-camera RGB video of a coloured tabletop object being pushed across the supplied calibration board. The data is used to recover the path geometry that becomes the robot policy prior.

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

ArUco centres define a planar homography. A configurable HSV mask is morphologically opened/closed; the largest valid contour supplies its centroid and a confidence based on area and circularity. Missing samples are interpolated, coordinates are smoothed, stationary lead-in/out is cropped, and the sequence is resampled with a shape-preserving cubic interpolator.

## Limitations

- Colour segmentation assumes a controlled scene and can select distractors.
- A single homography assumes a planar workspace and a fixed camera.
- Confidence measures detector quality, not semantic correctness.
- One demonstration is evidence for one task, not a general manipulation dataset.
- The public synthetic artifact exists only for software verification and is tagged `synthetic: true` throughout the pipeline.
