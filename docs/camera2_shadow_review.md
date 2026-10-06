# Camera 2 Shadow Review

Camera 2 has no tracking ground truth. This checklist is a short deployment safeguard for
the locked CrowdHuman candidate, not a replacement for an annotated benchmark.

## Artifacts

- Full comparison: `data/outputs/crowdhuman_a/camera-2-shadow/2_side_by_side.mp4`
- Contact sheet: `data/outputs/crowdhuman_a/camera-2-shadow/2_contact_sheet.jpg`
- Baseline video: `data/outputs/crowdhuman_a/camera-2-shadow/2_baseline.mp4`
- Candidate video: `data/outputs/crowdhuman_a/camera-2-shadow/2_crowdhuman_a.mp4`
- Diagnostic report: `data/outputs/crowdhuman_a/camera-2-shadow/report.json`
- Duration: 13.96 seconds (349 frames at 25 FPS)
- Baseline: pretrained YOLO26n at confidence 0.30
- Candidate: CrowdHuman fine-tuned checkpoint at confidence 0.55

The expected side-by-side SHA-256 is
`1436cee0b18835dd255f4ec45c5acf002a96ad31d95b8ecd62bc91bf32fec421`.

## Automated Diagnostics

| Model | Tracks | Avg active | Churn/1k observations | Max gap | E2E FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 59 | 10.415 | 16.231 | 72 | 13.769 |
| Candidate | 40 | 12.003 | 9.549 | 52 | 17.069 |

Lower churn and fewer total track IDs are consistent with improved stability, but these
values cannot determine whether a box is correct without ground truth.

## Manual Check

Watch the entire side-by-side video at normal speed, then inspect any uncertain interval
frame by frame. Accept the candidate only if all statements below are true:

- [x] Candidate-only boxes correspond to people rather than signs, reflections, luggage, or background texture.
- [x] The candidate does not introduce a persistent duplicate track on the same person.
- [x] The candidate does not miss an obvious person that the baseline tracks reliably.
- [x] Track IDs do not reset or transfer between nearby people more often than in the baseline.
- [x] Boxes remain usable on small, occluded, and edge-of-frame pedestrians.
- [x] No safety-relevant region is visibly worse with the candidate.

Record any problem with a timestamp and category (`false positive`, `miss`, `duplicate`,
`ID switch`, or `poor box`) before making the decision.

## Decision

- Review status: completed
- Reviewer: Khanh
- Decision: accept
- Notes: One ID transfer to the wrong person was observed at approximately 00:06-00:07
  while the people were very small in the frame. The reviewer accepted this residual issue.

Following this accepted review, the CrowdHuman checkpoint and confidence 0.55 were promoted
to `configs/person_tracking_final.yaml` without changing tracker, event, or output settings.
