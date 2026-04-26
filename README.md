# Aura Score Analyzer

Aura Score Analyzer is a Google Colab AI project with a simple website interface.

The app analyzes one uploaded photo using two approaches:

1. A transparent score made from three feature models:
   - Closed arms
   - Serious face
   - Glasses
2. A general black-box Aura model:
   - Aura
   - Not aura

The goal is to show why transparency matters in AI. The transparent program explains which features increased the score, while the general model gives an Aura probability without explaining the visual reason.

## Project Structure

```text
Aura Score Analyzer/
+-- aura_score_analyzer_colab.ipynb
+-- Data/
+-- Feature Detection Models/
|   +-- crossed_open_arms.zip
|   +-- glasses_no_glasses.zip
|   +-- smile_not_smile_model.zip
+-- general_model.zip
+-- requirements.txt
+-- README.md
```

## Open In Colab

After this folder is uploaded to GitHub, open:

```text
https://colab.research.google.com/github/mhirez/Aura-Score-Analyzer/blob/main/aura_score_analyzer_colab.ipynb
```

If your GitHub username or repository name is different, replace `mhirez/Aura-Score-Analyzer` in that link.

## How To Run

1. Open the notebook in Google Colab.
2. Run Cell 1 to install packages.
3. Run Cell 2 to clone this GitHub repo into Colab.
4. Run Cell 3 to extract the model ZIP files.
5. Run Cells 4, 5, and 6 once to prepare helpers and load the models.
6. Run Cell 8 to launch the website interface.
7. Upload a photo in the website and click Analyze Aura.

You can also use Cell 7 instead of the website if you want a simple notebook upload cell.

## Important

The notebook expects this repository to contain:

- `Feature Detection Models/crossed_open_arms.zip`
- `Feature Detection Models/smile_not_smile_model.zip`
- `Feature Detection Models/glasses_no_glasses.zip`
- `general_model.zip`
- `Data/`

The repository should be public if you want Colab to clone it without a password or token.
