# Dataset Notes

## Summary

This repository contains both model artifacts and image data used for training, testing, or demonstration. The current project structure is convenient for reproducibility, but the data provenance should be treated carefully.

## Repository Data Layout

The main data-bearing directories and files currently include:

- `Data/`
- `Feature Detection Models/`
- `general_model.zip`

These assets support:

- closed-arms detection
- serious-face detection
- glasses detection
- general Aura classification

## Provenance Status

The repository appears to contain a mix of:

- custom example photos
- project-specific training assets
- public-style face-image datasets or derived subsets

A full provenance, consent, and licensing audit for every image and model artifact is not yet captured in the repository in machine-readable form.

That means contributors and downstream users should not assume that every dataset or binary asset is automatically cleared for unrestricted redistribution or commercial use.

## Recommended Usage Guidance

- Treat the current data assets as research/demo materials unless rights are explicitly documented.
- Verify that you have permission before redistributing raw images or trained weights.
- If you plan to deploy the project publicly, prefer hosting large datasets and model binaries through a dedicated artifact store with clear metadata.
- If you collect new face images, obtain consent and document the collection policy.

## Privacy Considerations

Because the project processes face images and pose cues:

- avoid uploading private or sensitive images without consent
- avoid storing user uploads longer than necessary
- prefer ephemeral processing for demos
- remove personal identifiers from any newly added example data when possible

## Bias And Coverage Risks

Possible dataset limitations include:

- uneven demographic representation
- limited pose diversity
- limited lighting and camera diversity
- cultural bias in what is treated as "serious" or "aura-like"
- fashion/accessory bias in glasses detection

These limitations can affect both the transparent feature detectors and the general Aura model.

## Best Practice For Future Dataset Updates

For any new dataset or replacement asset, document:

- source
- date added
- license or permission status
- intended split or usage
- preprocessing performed
- whether the asset can be redistributed publicly

## Suggested Future Improvement

For a more production-ready repository, consider replacing raw in-repo dataset storage with:

- Git LFS
- GitHub Releases
- Hugging Face datasets or model storage
- cloud-hosted downloads with checksum verification

This keeps the code repository lighter and makes asset governance easier to track.
