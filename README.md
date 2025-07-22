# My House Price Prediction and Market Segmentation Project (Kuala Lumpur)

## Project Overview

This project focuses on **predicting housing prices** and **identifying distinct market segments** within Kuala Lumpur, Malaysia. Leveraging a dataset of various housing properties, I aim to build a robust regression model to estimate property values and then use clustering techniques to uncover natural groupings of properties, providing insights into different market niches.

## Objectives

* To develop a machine learning model capable of accurately predicting house prices in Kuala Lumpur.
* To perform market segmentation, identifying distinct clusters of properties based on their characteristics and potentially price trends.
* To gain insights into key factors influencing house prices and market dynamics in the region.

## Key Features & Components

* **Robust Data Preprocessing**: Comprehensive handling of missing values, inconsistent formats (e.g., in 'Size' and 'Price' columns), and data type conversions.
* **Feature Engineering**: Extraction of numerical features from text fields (e.g., converting 'Size' to square footage).
* **Regression Modeling**: Utilization of the **Random Forest Regressor** for predicting house prices.
* **Model Evaluation**: Assessment of the prediction model's performance using metrics like Mean Squared Error (MSE), R-squared (R2 Score), and Mean Absolute Error (MAE).
* **Unsupervised Learning for Segmentation**: Application of **K-Means Clustering** to identify market segments.
* **Cluster Analysis**: Detailed examination of the characteristics of each identified market segment (e.g., average price, dominant property types, common locations).
* **Dimensionality Reduction**: Use of **PCA (Principal Component Analysis)** for visualizing clusters in a lower-dimensional space.
* **Clustering Evaluation**: Assessment of segmentation quality using the Silhouette Score.

## Dataset

The project utilizes a `housing_data.csv` dataset, which contains various attributes of residential properties in Kuala Lumpur.

Key columns in the dataset include:
* `Price`: The target variable for price prediction.
* `Rooms`, `Bathrooms`, `Car Parks`: Number of rooms, bathrooms, and car park spaces.
* `Property Type`: Categorical description of the property (e.g., Apartment, Condominium, Terrace House).
* `Size`: Property size, often in varied string formats.
* `Furnishing`: Level of furnishing.
* `Location`: Geographical area of the property.
* Other features like `Land area`, `Tenure`, `Facing`, `Facilities`, etc.

## Project Structure

My project notebook (`HOUSING_PRICE_PREDICTION_AND_MARKET_SEGMENTATION_IN_KUALA_LUMPURE.ipynb`) is structured into logical sections, following a typical machine learning workflow.

1.  **Importing Dependencies**: Loading all necessary Python libraries.
2.  **Data Cleaning & Preprocessing**: Handling raw data inconsistencies and preparing features.
3.  **Exploratory Data Analysis (EDA)**: (Heading present, implying this is where further data insights would be explored if not directly in code snippets).
4.  **Random Forest Model for Price Prediction**: Training and evaluating the regression model.
5.  **Market Segmentation (Clustering)**: Performing clustering and analyzing segments.
6.  **Model Evaluation**: (Specific evaluation for clustering, using Silhouette Score).

## Setup & How to Run

This project is designed to be run in **Google Colab**, making it easy to set up and execute without local environment configurations.

### Prerequisites

* A Google account to access Google Colab.
* My `data_kaggle.csv` dataset. I recommend uploading this file directly to your Colab session or connecting your Google Drive.

### Steps to Run

1.  **Upload the Notebook to Google Colab**:
    * Go to [Google Colab](https://colab.research.google.com/).
    * Click on `File` > `Upload notebook` and select `HOUSING_PRICE_PREDICTION_AND_MARKET_SEGMENTATION_IN_KUALA_LUMPURE.ipynb`.
2.  **Upload Your Dataset**:
    * Once the notebook is open, click the "Files" icon on the left sidebar.
    * Click the "Upload to session storage" icon (folder with an arrow pointing up) and upload your `housing_data.csv` file. Ensure it's in the `/content/` directory or adjust the path in the notebook if you place it elsewhere (e.g., Google Drive).
    * *(Alternatively, if your data is in Google Drive, you'll need to mount your Drive first using `from google.colab import drive; drive.mount('/content/drive')` and then adjust the data loading path to `'/content/drive/My Drive/path/to/housing_data.csv'`)*.
3.  **Execute Cells Sequentially**:
    * Run each code cell in the notebook from top to bottom.
    * Observe the output for each step, including data cleaning progress, model performance metrics, and cluster analysis insights.

## Detailed Project Sections

### **1. Importing Dependencies**

* This initial section ensures all necessary Python libraries for data manipulation, visualization, machine learning, and regular expressions are loaded. I'm using `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (for `RandomForestRegressor`, `StandardScaler`, `KMeans`, `PCA`, `LabelEncoder`, and metrics), and `re`.

### **2. Data Cleaning & Preprocessing**

* **Initial Data Overview**: I start by reviewing the data types, checking for missing values (`.info()`, `.isnull().sum()`), and getting descriptive statistics (`.describe()`).
* **Cleaning 'Size' Column**: This was a complex step where I used regular expressions to extract numerical square footage from various string formats (e.g., "Built-up : 1,335 sq. ft.", "22 x 80 sq. ft.") and converted it to a consistent numeric type.
* **Cleaning 'Price' Column**: I removed currency symbols (`RM`), commas, and converted the column to a numeric (float) type. Rows with missing prices were dropped as 'Price' is my target.
* **Cleaning 'Rooms', 'Bathrooms', 'Car Parks'**: These columns contained non-numeric entries like "n.a." or "N.A.". I replaced these with `NaN` and imputed them (e.g., with median or 0), then converted them to integer types.
* **Handling Categorical Missing Values**: Missing entries in 'Furnishing' and 'Property Type' were filled with their respective modes.
* **Final Data Type Conversions**: Ensuring all relevant columns are in the correct numerical format for modeling.

### **3. Exploratory Data Analysis (EDA)**

* While this section heading is present, the provided notebook snippet doesn't explicitly contain new EDA visualizations. In a typical workflow, this phase would involve:
    * **Distribution Plots**: Histograms for numerical features, count plots for categorical features.
    * **Outlier Detection**: Box plots for numerical features.
    * **Correlation Analysis**: Heatmaps to understand relationships between numerical features and the target.
    * **Feature-Target Relationships**: Scatter plots or box plots comparing features against 'Price'.

### **4. Random Forest Model for Price Prediction**

* **Feature Selection**: I select the cleaned numerical and encoded categorical features for my regression model.
* **Data Splitting**: The dataset is split into training and testing sets to evaluate the model's generalization ability.
* **Model Training**: I use `RandomForestRegressor` due to its robustness and good performance on varied datasets.
* **Model Evaluation**: I assess the model's predictive accuracy using:
    * **Mean Squared Error (MSE)**: Measures the average squared difference between estimated values and actual value.
    * **R-squared (R2 Score)**: Represents the proportion of variance in the dependent variable that is predictable from the independent variables.
    * **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of forecasts, without considering their direction.

### **5. Market Segmentation (Clustering)**

* **Feature Selection for Clustering**: I carefully choose relevant features that define property characteristics for segmentation.
* **Feature Scaling**: I apply `StandardScaler` to these features, which is crucial for distance-based algorithms like K-Means.
* **K-Means Clustering**: I use the K-Means algorithm to group properties into distinct clusters. I initially explored `k=2` clusters.
* **Dimensionality Reduction (PCA)**: I apply PCA to reduce the dimensionality of my features, enabling me to visualize the clusters effectively.
* **Cluster Analysis**: For each identified cluster, I analyze its defining characteristics by looking at:
    * Average price.
    * Distribution of property types.
    * Dominant locations.
    * Average rooms, bathrooms, size, etc.
* **Cluster Visualization**: I use scatter plots (often with PCA components) to visualize the separation of the clusters.

### **6. Model Evaluation (Clustering Specific)**

* **Silhouette Score**: I evaluate the quality of my clustering solution using the Silhouette Score. This metric measures how similar an object is to its own cluster compared to other clusters, providing an indication of cluster separation. A higher score indicates better-defined clusters.

## Results & Insights

*(This section will be populated based on the actual output you get from running the notebook. Here are some examples of what you might include:)*

* **Price Prediction Performance**: My Random Forest model achieved an R2 score of [X.XX], an MSE of [Y.YY], and an MAE of [Z.ZZ] on the test set, indicating [e.g., "a good fit", "reasonable prediction accuracy"].
* **Key Price Drivers**: (Based on feature importances from Random Forest if you extract them, e.g., "Property size and location appear to be the most significant factors influencing housing prices.")
* **Identified Market Segments**: I identified [Number] distinct market segments.
    * **Cluster 1 (e.g., "Luxury Condos")**: Characterized by [high prices, large sizes, specific upscale locations like KLCC], predominantly [Condominiums].
    * **Cluster 2 (e.g., "Affordable Apartments")**: Characterized by [lower prices, smaller sizes, suburban locations like Cheras], predominantly [Apartments].
    * **Cluster 3 (e.g., "Family Terrace Houses")**: Characterized by [mid-range prices, moderate sizes, terrace house type, family-friendly locations].
* **Segmentation Quality**: The Silhouette Score of [X.YY] suggests that my clusters are [well-separated / moderately separated / need further refinement].

## Future Work & Improvements

* **Hyperparameter Tuning**: Optimize the `RandomForestRegressor` and `KMeans` parameters using techniques like GridSearchCV or RandomizedSearchCV.
* **More Advanced Regression Models**: Explore other regression algorithms such as XGBoost, LightGBM, or CatBoost, which often perform well on tabular data.
* **Optimal Number of Clusters (Elbow Method/DBSCAN)**: Systematically determine the optimal 'k' for K-Means using methods like the Elbow Method or gap statistic, or explore density-based clustering like DBSCAN.
* **Outlier Treatment**: Implement more sophisticated outlier detection and handling techniques.
* **Time Series Analysis**: If date information (e.g., listing date, transaction date) were available, explore time series forecasting for price trends.
* **Geospatial Analysis**: Integrate geospatial data (e.g., proximity to amenities, public transport) for richer feature engineering and location-based insights.
* **Deployment**: Consider deploying the price prediction model as a web service.

## Technologies Used

* Python 3.x
* Pandas (Data manipulation and analysis)
* NumPy (Numerical operations)
* Matplotlib (Plotting)
* Seaborn (Statistical data visualization)
* Scikit-learn (Machine learning models, preprocessing, and metrics)
* `re` (Regular expressions for string parsing)
* Google Colab (Development environment)

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details. *(Consider creating a separate `LICENSE` file in your GitHub repository)*.
