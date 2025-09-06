# 🎯 Customer Purchase Propensity Analysis

A comprehensive machine learning system that predicts customer purchase likelihood based on behavioral patterns and interaction data. This project combines advanced ML techniques with an interactive web dashboard for real-time customer analysis.

App link : https://22ckwyowfninwg3rnlahjs.streamlit.app/

<img width="1828" height="900" alt="Screenshot 2025-09-05 163750" src="https://github.com/user-attachments/assets/fab39a1d-0c90-4f87-bad7-e2f9ff2aa1a8" />
<img width="1855" height="904" alt="Screenshot 2025-09-05 163916" src="https://github.com/user-attachments/assets/9dee3a19-36f4-4d09-ba18-2be66c9f62b8" />



## 📊 Project Overview

**Customer Propensity Analysis** is an end-to-end solution that:
- Analyzes customer behavior patterns from e-commerce event data
- Builds predictive models using XGBoost and advanced feature engineering
- Provides real-time propensity scoring through an interactive Streamlit dashboard
- Handles severe class imbalance with scale_pos_weight parameter
- Achieves 98.20% accuracy with 0.9890 ROC AUC score

## 🏗️ Architecture

```
├── 📁 data/                    # Raw datasets
│   ├── events.csv             # User interaction events (90MB)
│   ├── item_properties_part1.csv  # Item metadata part 1 (462MB)
│   ├── item_properties_part2.csv  # Item metadata part 2 (390MB)
│   └── category_tree.csv      # Product category hierarchy (14KB)
├── 🔧 pipeline.py             # Feature engineering pipeline
├── 🤖 customer_propensity.ipynb  # Complete ML analysis notebook
├── 🌐 app.py                  # Streamlit web application
├── 📦 propensity_model.pkl    # Trained XGBoost model
├── 📊 visitor_df_final.csv    # Processed features dataset (74MB)
├── 🐳 Dockerfile              # Container configuration
└── 📋 requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB+ RAM (for data processing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-propensity
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

```bash
# Build the container
docker build -t customer-propensity .

# Run the container
docker run -p 8501:8501 customer-propensity
```

## 📈 Data Pipeline

### Feature Engineering

The system extracts 7 key behavioral features:

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `view_count` | Total item views | Count of 'view' events |
| `addtocart_count` | Cart additions | Count of 'addtocart' events |
| `unique_items_viewed` | Distinct products | Unique item IDs viewed |
| `add_to_cart_rate` | Conversion rate | addtocart_count / view_count |
| `recency_days` | Days since last activity | Current date - last event date |
| `num_sessions` | Browsing sessions | Sessions with 30-min timeout |
| `avg_events_per_session` | Engagement intensity | Total events / session count |

### Data Processing Steps

1. **Data Loading**: Load events, properties, and category data
2. **Cleaning**: Remove duplicates, handle missing values
3. **Feature Extraction**: Calculate behavioral metrics per visitor
4. **Target Creation**: Binary classification (purchased = 1, not = 0)
5. **Class Imbalance**: Handle with XGBoost scale_pos_weight parameter

## 🤖 Machine Learning Model

### Model Selection

- **Final Algorithm**: XGBoost Classifier with hyperparameter tuning
- **Hyperparameter Tuning**: Optuna + RandomizedSearchCV
- **Class Imbalance**: Scale_pos_weight parameter (not SMOTE)
- **Cross-validation**: 3-fold CV with F1-weighted scoring

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.20% |
| **ROC AUC** | 0.9890 |
| **Precision (Class 1)** | 30.14% |
| **Recall (Class 1)** | 93.52% |
| **F1-Score (Class 1)** | 0.46 |
| **Matthews Correlation** | 0.5194 |

### Model Comparison Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| **LightGBM** | 97% | 0.67 | Good baseline |
| **XGBoost** | 98% | 0.71 | Better performance |
| **XGBoost + Hyperparameter Tuning** | **98.20%** | **0.72** | **Best overall** |

## 🌐 Web Application

### Dashboard Features

- **Interactive Interface**: Real-time customer analysis
- **Pre-built Profiles**: High/Medium/Low propensity examples
- **Dynamic Scoring**: Live propensity calculation
- **Visual Analytics**: Plotly-powered charts
- **Responsive Design**: Professional dashboard layout

### Usage

1. **Select User Profile**: Choose from example visitor types
2. **Analyze Customer**: Click "Analyze Customer" button
3. **View Results**: See propensity score and key metrics
4. **Explore Insights**: Analyze behavioral patterns

## 📊 Data Insights

### Dataset Statistics

- **Total Events**: 2,755,641 interactions
- **Unique Visitors**: 1,406,580 users
- **Products**: 417,053 unique items
- **Categories**: 1,669 category nodes
- **Time Period**: June-September 2015

### Class Distribution

- **Non-purchasers**: 1,395,861 (99.17%)
- **Purchasers**: 11,719 (0.83%)
- **Imbalance Ratio**: 119:1

### Key Behavioral Patterns

- **Session Timeout**: 30 minutes of inactivity
- **High Engagement**: Users with >10 views show higher purchase rates
- **Cart Conversion**: Strong correlation between add-to-cart and purchase
- **Recency Impact**: Recent activity strongly predicts purchase likelihood

## 🔧 Technical Details

### Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
streamlit>=1.20.0
joblib>=1.1.0

```

### Performance Optimizations

- **Caching**: Streamlit caching for data and model loading
- **Efficient Processing**: Vectorized pandas operations
- **Memory Management**: Optimized data types and chunked processing
- **Model Serialization**: Fast inference with joblib

### Code Quality

- **Modular Design**: Separated concerns (pipeline, app, analysis)
- **Error Handling**: Graceful handling of edge cases
- **Documentation**: Comprehensive inline comments
- **Testing**: Validation of feature engineering pipeline

## 📁 File Descriptions

### Core Files

- **`pipeline.py`**: Feature engineering functions for real-time visitor analysis
- **`app.py`**: Streamlit web application with interactive dashboard
- **`customer_propensity.ipynb`**: Complete Jupyter notebook with analysis, model training, and evaluation

### Data Files

- **`events.csv`**: Raw user interaction events with timestamps
- **`item_properties_*.csv`**: Product metadata and attributes
- **`category_tree.csv`**: Hierarchical product categorization
- **`visitor_df_final.csv`**: Processed features for all visitors

### Model & Output

- **`propensity_model.pkl`**: Serialized XGBoost model for production use
- **`visualizations/`**: Complete model evaluation visualizations (7 PNG files)

## 🚀 Deployment

### Production Considerations

- **Scalability**: Model can handle real-time scoring
- **Monitoring**: Log propensity scores and predictions
- **Updates**: Retrain model periodically with new data
- **Security**: Implement authentication for dashboard access

### Environment Variables

```bash
# Optional: Set for production
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Data Source**: Kaggle e-commerce dataset
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact: [Your Contact Information]
- Documentation: [Link to detailed docs]

---

**Built with ❤️ for customer behavior analysis and predictive modeling**
