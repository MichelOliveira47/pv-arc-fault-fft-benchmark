# Contributing

Thank you for helping improve reproducibility.

## Report a problem

Use the bug-report template and include:

- operating system and Python version;
- package versions (`python -m pip freeze`);
- run mode and `PV_ARC_*` settings;
- the failing pipeline step and full traceback;
- whether the input validator passed.

Do not upload restricted data, trained models derived from restricted data, credentials, or private institutional material to an issue.

## Propose a change

1. Create a focused branch.
2. Keep data and generated outputs outside Git history.
3. Run `python -m unittest discover -s tests -v`.
4. Run the quick check when the labeled input is available.
5. Explain any change that could alter feature extraction, temporal splitting, model selection, or reported metrics.

Changes affecting the scientific workflow should preserve the original configuration as the default and expose alternatives through clearly documented switches.
