# Experiment report

Data source: **personal recording**
SHA-256: `e6a12a9907155972615d5e0e45dcf46ebf3d3053fc088b32db35ec9b5d34b52d`
Valid colour detections: 100.0%
Per-frame marker calibration: 100.0%
Evaluation domains: 20

| Controller | Success rate | Final error (mean ± std) | Mean pusher clearance |
|---|---:|---:|---:|
| Straight | 5% | 14.0 ± 2.5 cm | -0.1 cm |
| Video Prior | 100% | 5.7 ± 0.3 cm | 4.1 cm |
| Optimized | 100% | 5.2 ± 0.3 cm | 2.2 cm |

CEM best objective: `0.9962`.

The randomized evaluation changes puck mass, friction, start pose, actuator gain, and joint damping for every seed. The video-prior controller retains the recorded path; the optimized controller may only add a smooth, bounded residual.
