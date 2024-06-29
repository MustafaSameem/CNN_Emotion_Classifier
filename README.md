# OB_02 Project

## Group Members
- Mustafa Sameem
- Mehdi Fouzail
- Nabih El-helou

## File/Folder Information

### imageClass Folder
This folder contains all 2000 images, organized into 4 .zip files.

### imageSamples Folder
This folder contains around 25 sample pictures of each class.

### Python Files
- **distribution.py**: Generates the class distribution chart.
- **pixel.py**: Generates the pixel intensity distribution chart.
- **sample.py**: Generates sample images along with their histogram charts.
- **brightness.py**: Increases brightness for darker images.

## Executing Data Cleaning

In order to execute `brightness.py`, simply `cd` into the directory containing the file and modify the input string to the desired class ("Angry", "Neutral", "Engaged", "Happy") you want to apply the changes on.

## Executing Data Visualization

In order to visualize the data using `distribution.py`, `pixel.py`, and `sample.py`, simply `cd` into the directory containing the respective file and run the file code. Relative paths are included to ensure cross-platform runnability.

## Executing Phase II

1. Recreate the conda environment using the provided `environment.yml` file.

2. Navigate to the `Phase2` directory.

3. Run the following commands:
   - `python cnn.py` to run the main model.
   - `python cnn_variant1.py` to run the variant1 model.
   - `python cnn_variant2.py` to run the variant2 model.

To evaluate the models, execute the following commands:
- `python evaluate_model.py` for the main CNN model.
- `python evaluate_variant1.py` for the variant1 model.
- `python evaluate_variant2.py` for the variant2 model.


## Executing Phase III

1. Recreate the conda environment using the provided `environment.yml` file.

2. Navigate to the `Phase3` directory.

3. Run the following commands:
   - `python Kfold.py` to run the kfold model.
   - `python cnn_male.py` to run the male group model.
   - `python cnn_female.py` to run the female group model.

To evaluate the models, execute the following commands:
- `python evaluate_model_male.py` for the male model measures.
- `python evaluate_model_female.py` for the female model measures.

[GitHub Repository](https://github.com/MustafaSameem/COMP472)
