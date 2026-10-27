# Safety specification v3: switching-centered evidence review

Reviewed 2026-10-27. This is a SIL research specification, not a certificate or a regulatory compliance test.
The 24 supplied sources are registered individually in `pipeline/09_monitor/safety_spec.yaml.references` with retrieval status and permitted use. Unretrieved documents do not supply numerical thresholds.

## What the evidence changes

1. Separate safety constraints, controller-switch requirements, tracking performance and comfort.
   [Dauner et al. supplementary, §2.2](https://www.cvlibs.net/publications/Dauner2023CORL_supplementary.pdf)
   treats collisions/drivable area and comfort as different metric classes. Full-route scores do not replace local handover checks.
2. Retain 60°/s switch-transient and 90°/s command-bound values for experiment continuity, but label them design/configuration values. The supplied lateral-control comparison and multi-tier steering papers motivate rate/smoothness measurement, not universal numeric limits.
3. [TreeIRL Appendix B.1](https://arxiv.org/html/2509.13579v4) reports empirical comfort bounds. Adopt only three dimensionally compatible references: lateral acceleration 4.89 m/s², yaw rate 0.95 rad/s and yaw acceleration 1.93 rad/s². These are report-only. Do not mix the total jerk norm with lateral jerk or claim official nuPlan signal processing.
4. [SAE 2017-01-1560](https://saemobilus.sae.org/papers/vehicle-stability-criterion-research-based-phase-plane-method-2017-01-1560) describes state-dependent beta/yaw stability regions. No validated region is provided for our vehicle/ODD, so SR3 remains unassessed. Raw beta/yaw measurements are not proof that LQR can recover.
5. TLC predicts boundary crossing under assumptions. Our local margin/closing-rate diagnostic is explicitly an approximation. It is not the full Mammar model and never replaces measured no-reaction FTTI. No 1.5 s or other universal TLC pass threshold is adopted.
6. [Euro NCAP LSS v4.3](https://cdn.euroncap.com/cars/assets/euro_ncap_lss_test_protocol_v43_f2ddd5f6d6.pdf) §4.1.1 requires at least 100 Hz dynamic-data acquisition. Our 20 Hz default is not equivalent. Its tyre-based DTLE must not be confused with our conservative rectangular-body lane margin. The original supplied media URL returned 404; the official protocol index supplies the replacement link.
7. Official ISO 11270 and 22735 scope pages were accessible; complete numeric clauses were not. No numerical ISO compliance claim is made. The supplied ISO 2631 secondary paper and SKKU page were not retrieved. ISO 2631 frequency weighting/exposure processing is not implemented.
8. The UN R79-linked patent is design prior art, not a regulation. The linked amendment index concerns sampling/filtering/measurement position; its UNECE PDF returned 403. Neither justifies treating 4.41 m/s² as a universal R79 safety bound.
9. RSS/Safe Swerve need actor-relative state, geometry and response assumptions; those checks are deferred rather than approximated by an arbitrary TTC value. Motion-sickness-study delta-acceleration thresholds are not transplanted into instantaneous ay checks.
10. STL robustness motivates signed margins, but per-run normalized min margins are not population risk, CVaR or a proof over continuous trajectories.

## Implementation changes tied to this review

- `safety_spec.yaml`: v3 taxonomy, threshold provenance, source registry, scope and deferred checks.
- `performance.py`: switch-aligned numerical diagnostics; baseline aligned to the fault handover; missing/low-speed/unbounded predictions remain null with reasons.
- `monitor.py`: reads reference limits; records spec SHA-256; actually uses configured P02 lateral-acceleration bound; includes enforced SR6 in failure/overall; restricts recovery hold and transition checks to the evaluated window; never rounds a sub-tick FTTI budget upward; validates optional channels; records mandatory-measurement completeness.
- `dashboard.py`: uses returned threshold values, reports incomplete assessments and exposes diagnostic phase metrics separately.
- `explain.py` and `feedback.py`: preserve diagnostics for insight analysis and exclude incomplete safety assessments from complete-data analysis.

## Experimental protocol

Freeze criteria before observing campaign results. Record source commit, simulator/controller versions, seed, dt, spec SHA-256 and resolved overrides. Use matched baseline/fault/no-reaction configurations. Keep safety labels separate from comfort/accuracy hypotheses. Compare paired differences and multiple independently varied seeds; repeated synthetic traces are only a display/test fixture.

Threshold choices require vehicle and ODD calibration before stronger claims: road-wheel versus steering-wheel units, actuator/command bounds, lane geometry, tyre friction, stability region and signal processing must be explicit. The current 0.5 FTTI fraction, 0.5 s fallback, 0.5 s transition window, 5 s post window and recovery bands remain design choices.

## Open items

- Obtain full SKKU/HAL/SCITEPRESS and numeric ISO clauses if they are to support thresholds.
- Add actor logging and a validated RSS/TTC implementation before scenario-relative collision-risk assertions.
- Supply validated stability/recoverability artifacts for SR3.
- Add higher-rate logging and protocol-specific filters before Euro NCAP/UN R79 comparisons.
- Add exposure/frequency weighting before ISO 2631 comfort assessment.
