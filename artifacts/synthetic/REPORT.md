# Experiment report

Data source: **synthetic smoke test**
SHA-256: `9de6a959eb1a2eb46378a3775868277676e1f00791a13177f1199ec392231b77`
Valid colour detections: 100.0%

| Controller | Success rate | Final error (mean ± std) | Mean clearance |
|---|---:|---:|---:|
| Straight | 0% | 17.6 ± 3.3 cm | -0.1 cm |
| Video Prior | 100% | 3.1 ± 0.6 cm | 8.1 cm |
| Optimized | 100% | 2.8 ± 0.7 cm | 5.2 cm |

CEM best objective: `0.4792`.

The randomized evaluation changes puck mass, friction, start pose, actuator gain, and joint damping for every seed. The video-prior controller retains the recorded path; the optimized controller may only add a smooth, bounded residual.
