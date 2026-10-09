# Personal capture protocol

This protocol is intentionally short enough to repeat if tracking fails.

## Set up

- Print `artifacts/capture_board.pdf` as A4 landscape at 100%; disable "fit to page".
- Measure the printed distance between marker 0 and marker 1 centres. It should be 247 mm (a few millimetres of printer error is acceptable).
- Use a round object with a vivid magenta/pink top. A coloured sticker on a bottle cap is enough.
- Use diffuse light. Avoid a magenta sleeve, mat, or reflection in the calibrated rectangle.
- Mount the phone approximately 40–60 cm above the page. Perspective is acceptable; camera motion during the take is not.

## Record

1. Keep the object stationary for roughly one second.
2. Push it in one continuous motion. If an obstacle is used, make the route intentional rather than tracing the shortest path.
3. Keep the object stationary at the goal for roughly one second.
4. Keep all four corner markers visible for the whole take.

1080p/30 fps is ample. A 10–20 second clip is easier to inspect than a long recording.

## Validate before running MuJoCo

Run only extraction first:

```bash
trace2touch extract \
  --config configs/personal.yaml \
  --video data/raw/my_push.mp4 \
  --output runs/personal/demonstration.npz \
  --overlay runs/personal/tracking_overlay.gif
```

Inspect the overlay. The green circle must remain on the object. The command prints `valid_detection_fraction`; repeat the take or adjust `object_hsv_ranges` when it is below 0.9. Set `object_radius_m` to enable metric size, board-bound, and temporal-continuity rejection of colour distractors.

## Public-data check

Raw recordings under `data/raw/` are ignored by default. A recording should only be placed under `data/public/` after the collector decides that its audio, background, and camera metadata are suitable for publication. Never relabel the synthetic smoke test as personal data. Keep the derived `.npz`, the tracking overlay, and the source SHA-256 in the public result so a reviewer can see exactly how the recording entered the system.
