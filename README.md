# Parking Availability Prediction

This repository contains a complete pipeline for predicting parking availability using a combination of historical data, weather data, and road features. The project involves data preprocessing, feature engineering, and building a Random Forest regression model to make predictions.

---

## Table of Contents
- [Datasets](#datasets)
- [Steps in the Pipeline](#steps-in-the-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)

---

## Datasets
This project uses the following datasets:
1. **Ground Truth Data:** Contains parking occupancy data with `road_segment_id`, `timestamp`, `max_capacity`, `occupied`, and `available`.
2. **Weather Features:** Includes weather data like temperature (`tempC`), windspeed (`windspeedKmph`), and precipitation (`precipMM`).
3. **Road Features:** Provides features about the area around the road segment such as commercial zones, residential zones, schools, event sites, and shopping areas.

---

## Steps in the Pipeline
1. **Load and Merge Datasets:**
   - The datasets are loaded using pandas and merged on `road_segment_id` and `timestamp`.
2. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Extract useful features like year, month, day, and hour from the timestamp.
   - Apply skewness correction and standardization to numeric columns.
3. **Feature Engineering:**
   - Remove unnecessary columns and handle duplicates.
   - Transform numeric features to improve the model’s performance.
4. **Model Training and Hyperparameter Tuning:**
   - Train a Random Forest regression model to predict parking availability.
   - Use RandomizedSearchCV for hyperparameter optimization.

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/parking-availability-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd parking-availability-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage
1. Update the dataset paths in the code to match your local environment.
2. Run the preprocessing script:
    ```bash
    python preprocess_data.py
    ```
3. Train the model:
    ```bash
    python train_model.py
    ```
4. Evaluate the model and view results.

---

## Features
- **Data Cleaning:** Handles missing values and outliers.
- **Feature Engineering:** Extracts meaningful date-time features and standardizes data.
- **Model Training:** Uses Random Forest regression with hyperparameter tuning for optimal results.

---

## Model Training and Evaluation
The Random Forest model is trained on the cleaned dataset. Evaluation metrics include:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

---

## Results
- The optimized model achieved the following evaluation metrics:
    - **Mean Absolute Error (MAE):** X.XX
    - **Root Mean Squared Error (RMSE):** X.XX
    - **R² Score:** X.XX
    - **Final Accuracy:** XX.XX%

---

## Contributing
Feel free to open issues or create pull requests to improve this project.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Acknowledgments
Special thanks to all contributors and open-source libraries used in this project.
