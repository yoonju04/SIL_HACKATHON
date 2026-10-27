# Nemotron Prior Knowledge Extension

Patch target: `yoonju04/SIL_HACKATHON`, branch `new_pipeline`. This is an ADDITIVE patch: original YAML pipeline files are preserved.

## Structure
- `pipeline/00_prior_knowledge/prior_knowledge.yaml`: physics relation and explicitly unverified hypotheses.
- `engine/prior_tools/prior_tools.py`: builds model-independent Nemotron message payload using *verified* safety-profile observations only; includes separate nominal lateral acceleration screening function.
- `tests/test_prior_tools.py`: lightweight smoke tests.

## Install and run
From repository root:

```bash
python -m pip install pyyaml
python -m engine.prior_tools.prior_tools --output nemotron_prior_payload.json
python -m unittest discover -s tests -p 'test_prior_tools.py' -v
```

`nemotron_prior_payload.json` is for inspecting messages before integrating with existing Nemotron API invocation. This patch does not send a network request, train a GP, enforce a full feasibility filter, or run CARLA.

## Important distinctions
- Actual vehicle/controller values must be evidenced by code observations (O-IDs); null/unchecked values are never promoted to verified facts.
- Priors PK-002/PK-003 are **test hypotheses**, not proven vulnerabilities.
- a_y = v² |curvature| is a steady-state demand estimate, not a full feasibility determination.
- Stress-test candidates should be tagged separately, not discarded as unsafe.
- This patch must be reviewed and integrated into stage 03/04 orchestration before it affects runtime search.
