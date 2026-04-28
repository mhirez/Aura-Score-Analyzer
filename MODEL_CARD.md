# Model Card

## Model Summary

Aura Score Analyzer is not a single model. It is a small inference system that combines several TensorFlow/Keras image classifiers into one explainable score.

The current project includes:

| Component | Purpose | Input | Output |
| --- | --- | --- | --- |
| Closed-arms model | Detect closed or crossed arms | Full image | Probability of the target pose |
| Serious-face model | Detect serious or non-smiling expression | Face crop | Probability of the target expression |
| Glasses model | Detect visible glasses | Face crop | Probability of glasses |
| General Aura model | Predict Aura vs not Aura directly | Full image | Aura probability |

## Intended Use

This repository is intended for:

- student ML projects
- explainable AI demonstrations
- image-classification workflow examples
- Colab and Streamlit demos

## Out-Of-Scope Use

This project is not designed for:

- hiring or admissions decisions
- surveillance or law-enforcement use
- psychological or medical interpretation
- attractiveness scoring
- any high-stakes judgment about a person

The term "Aura" in this repository is a playful project label, not a scientific construct.

## How The System Produces A Score

The transparent score is computed from three component probabilities:

```text
Aura Score = 100 * ((P(closed arms) + P(serious face) + P(glasses)) / 3)
```

Each component contributes equally. The separate general Aura model is presented only as a comparison point and not as the source of the final explainable score.

## Model Inputs

- RGB images
- face crop for the serious-face and glasses classifiers
- full image for the arm classifier and the general Aura model

The system uses OpenCV Haar Cascade face detection for face cropping, with a full-image fallback if no face is detected.

## Training And Export Format

The repository loads models from exported TensorFlow/Keras artifacts, including legacy `.h5`, `.keras`, `SavedModel`, and Teachable Machine-style layouts when available.

The current repo is optimized for inference, demonstration, and deployment. It does not yet include a complete training pipeline with tracked experiments, versioned metrics, or dataset manifests for every model artifact.

## Evaluation Status

The repository currently does not publish a standardized benchmark table for:

- accuracy
- precision / recall
- calibration
- subgroup fairness
- robustness across lighting, pose, or image quality

Results should therefore be treated as qualitative demo outputs rather than production-validated model claims.

## Known Limitations

- Face detection can fail on occlusions, unusual angles, or low-resolution images.
- The score depends strongly on how clearly the arms and face are visible.
- The models may be sensitive to lighting, cropping, camera distance, and background context.
- The general Aura model is less interpretable than the transparent score.
- The project may reflect dataset bias in pose, fashion, demographics, or photography style.

## Ethical Considerations

- The project analyzes face and pose cues, so privacy and consent matter.
- Outputs may encode social or dataset bias.
- The score can be misread as objective even though it is subjective and synthetic.
- The system should only be used in playful, educational, or exploratory settings.

## Maintenance Notes

When updating or replacing models, document:

- the source of the training data
- export format and TensorFlow version
- any label-map changes
- any score-formula changes

If a future version introduces formal evaluation metrics, add them here so readers can compare model revisions over time.
