# Release notes

This release provides fixed differential calibration for cautious natural language inference using local language models.

Included:

- Ordinary and two cautious prompt wordings, with complete and premise-removed input scoring.
- Fixed differential calibration, sample-specific correction and comparison baselines.
- Accuracy, macro-F1, class recall and paired premise-group bootstrap evaluation.
- Synthetic examples, dependency versions and CPU checks.
- Reference summaries for main accuracy results, calibration sensitivity, runtime and transfer evaluations, including unsuccessful settings.

The release contains no manuscript, model weights, benchmark text, exact benchmark selections or raw experimental predictions. It is a method implementation and result-summary release, not an end-to-end reproduction archive for every experiment. Model scoring requires a compatible CUDA GPU; calibration and metric computation use CPU NumPy.

The positive empirical evidence is specific to Qwen–SICK. Results should be validated for each intended model and task.
