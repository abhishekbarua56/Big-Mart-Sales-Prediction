# Big Mart Sales Prediction#

## Project Description

The **Big Mart Sales Prediction** project aims to build a predictive model that forecasts sales for various products in Big Mart outlets based on historical sales data. Understanding sales patterns is crucial for optimizing inventory management, designing effective marketing strategies, and enhancing customer satisfaction. This project employs machine learning techniques to analyze factors influencing sales, ultimately providing actionable insights to support strategic decision-making.

### Objectives:
- To predict future sales for Big Mart outlets using historical sales data.
- To identify key factors that significantly impact sales performance.
- To provide a data-driven approach for inventory management and marketing strategies.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Data Sources](#data-sources)
- [Key Steps and Methods](#key-steps-and-methods)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
- [Instructions to Run the Project](#instructions-to-run-the-project)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Technologies Used

- **Python**: The primary programming language used for data analysis and modeling.
- **Jupyter Notebook**: An interactive environment for running Python code and visualizing results.
- **Pandas**: A powerful library for data manipulation and analysis.
- **NumPy**: A library for numerical computations that complements Pandas.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations.
- **Seaborn**: A statistical data visualization library based on Matplotlib.
- **Scikit-learn**: A library providing simple and efficient tools for data mining and machine learning.
- **XGBoost**: An optimized gradient boosting framework designed to enhance model performance.

## Data Sources

The dataset used in this project consists of historical sales data from Big Mart, available on [Kaggle](https://www.kaggle.com/datasets/pritishn/big-mart-sales-prediction). It contains various features, including:

- **Item_Identifier**: Unique identifier for each item.
- **Item_Weight**: Weight of the item.
- **Item_Fat_Content**: Indicates whether the item is low-fat or regular.
- **Item_Visibility**: The percentage of total display area of a product allocated to the item.
- **Item_Type**: The category of the item.
- **Outlet_Identifier**: Unique identifier for each outlet.
- **Outlet_Establishment_Year**: The year the outlet was established.
- **Outlet_Size**: Size of the outlet.
- **Outlet_Location_Type**: The type of location of the outlet (urban, semi-urban, rural).
- **Outlet_Type**: Type of the outlet (e.g., grocery store, supermarket).
- **Sales**: The sales of the item at the outlet.

### Preprocessing Steps
- **Missing Value Handling**: Addressed missing values in `Item_Weight` and `Outlet_Size` using statistical methods and imputation.
- **Feature Engineering**: Created new features like `Item_Fat_Content` and grouped `Item_Visibility` into bins for better modeling.
- **Encoding Categorical Variables**: Used label encoding and one-hot encoding to convert categorical variables into numerical formats suitable for modeling.

## Key Steps and Methods

### Data Preprocessing
- Removed duplicates and irrelevant columns to ensure a clean dataset.
- Standardized and normalized the data where necessary to facilitate better model performance.
- Conducted exploratory analysis to visualize distributions and relationships among features.

### Exploratory Data Analysis (EDA)
- Utilized visualizations (bar plots, box plots, and histograms) to identify trends and correlations between features and sales.
- Investigated the impact of outlet type, location, and size on sales performance.
- Analyzed item visibility and its correlation with sales, revealing potential insights for marketing strategies.

### Model Building
- Trained several regression models, including:
  - **Linear Regression**: Established a baseline model for sales prediction.
  - **Ridge Regression**: Applied regularization to handle multicollinearity.
  - **Lasso Regression**: Used for feature selection through regularization.
  - **XGBoost**: Implemented an advanced boosting algorithm to improve predictive accuracy.

- Used **train-test split** to evaluate model performance on unseen data, ensuring a robust validation process.

### Model Evaluation
- Evaluated models using:
  - **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors.
  - **R-Squared**: Indicates the proportion of variance in the dependent variable predictable from the independent variables.
- Selected the XGBoost model as the final model, achieving superior performance with optimal hyperparameters.

## Instructions to Run the Project

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/abhishekbarua56/Big-Mart-Sales-Prediction
   ```

2. Navigate to the project directory:
   ```bash
   cd Big-Mart-Sales-Prediction
   ```

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook (`Final_Project_Big_Mart_Sales_Prediction.ipynb`) in Jupyter Lab or Jupyter Notebook and execute the cells in order.

## Usage Examples

To make predictions, you can input new data into the trained model. Here’s a basic example of how to use the model for predictions (example provided in the notebook):

```python
import pandas as pd

# Load the trained model
import joblib
model = joblib.load('xgboost_model.pkl')

# Prepare new data
new_data = pd.DataFrame({
    'Item_Weight': [15.0],
    'Item_Fat_Content': ['Low Fat'],
    'Item_Visibility': [0.05],
    'Item_Type': ['Dairy'],
    'Outlet_Identifier': ['OUT027'],
    'Outlet_Size': ['Medium'],
    'Outlet_Location_Type': ['Tier 1'],
    'Outlet_Type': ['Supermarket Type1'],
    'Outlet_Establishment_Year': [2005]
})

# Make predictions
predicted_sales = model.predict(new_data)
print(f'Predicted Sales: {predicted_sales}')
```

## Results

The final XGBoost model achieved an RMSE of **[insert RMSE value]** and an R-squared value of **[insert R-squared value]**, indicating a strong predictive capability. Key findings from the analysis included:
- **Top Factors Influencing Sales**: Outlet size and type were found to have a significant impact on sales performance.
- **Sales Trends**: Seasonal trends and customer preferences can be identified for targeted marketing.

Visualizations and detailed metrics can be found throughout the Jupyter Notebook.

## Future Improvements

- **Enhanced Feature Engineering**: Further feature creation based on external data sources, such as economic indicators or regional demographics, could improve predictions.
- **Advanced Modeling Techniques**: Experimenting with deep learning models or ensemble methods for more complex relationships.
- **Real-time Predictions**: Develop an API for real-time sales predictions based on incoming data.

## Contributing

Contributions are welcome! If you would like to contribute to this project:
1. Fork the repository.
2. Create a new branch (e.g., `feature-branch`).
3. Make your changes and commit them.
4. Submit a pull request detailing your changes and their significance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact Information

For questions, suggestions, or feedback, please feel free to reach out:

- **Name**: Abhishek Ranjit Barua
- **Email**: babi17no@gmail.com
- **GitHub**: [Abhishek's Profile](https://github.com/abhishekbarua56)
```

Feel free to replace the placeholder text for RMSE and R-squared values with actual results from your project. Let me know if you need any further modifications!
