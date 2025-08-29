# 🚀 Quick Start Guide - Customer Propensity Analysis

Get up and running with the Customer Propensity Analysis system in under 10 minutes!

## ⚡ Quick Start (3 Steps)

### 1. 🐍 Setup Environment
```bash
# Clone or download the project
cd customer-propensity

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 2. 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. 🚀 Run the Application
```bash
streamlit run app.py
```

**That's it!** 🎉 Your dashboard will open at `http://localhost:8501`

---

## 🎯 What You'll Get

### **Interactive Dashboard**
- Real-time customer propensity scoring
- Pre-built visitor profiles (High/Medium/Low propensity)
- Behavioral metrics and insights
- Professional, responsive design

### **Model Performance**
- **98.20% Accuracy** on customer purchase prediction
- **0.9890 ROC AUC** score
- Real-time scoring capabilities
- Comprehensive evaluation metrics

---

## 🔍 Explore the System

### **Dashboard Features**
1. **Select User Profile** from the sidebar
2. **Click "Analyze Customer"** button
3. **View Results**: Propensity score + key metrics
4. **Explore Insights**: Behavioral patterns analysis

### **Example Profiles**
- **High Propensity User**: Likely to purchase
- **Medium Propensity User**: Moderate purchase likelihood
- **Low Propensity User**: Unlikely to purchase

---

## 📊 Run Model Evaluation

Want to see detailed model performance metrics and visualizations?

### **Windows Users**
```bash
run_evaluation.bat
```

### **macOS/Linux Users**
```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

### **Manual Execution**
```bash
python model_evaluation.py
```

**This will generate:**
- 📂 `visualizations/` folder with all plots
- Confusion matrix visualization
- ROC curve analysis
- Precision-recall curves
- Feature importance plots
- Threshold analysis
- Class distribution charts
- Correlation heatmaps

---

## 🐳 Docker Deployment

### **Build & Run**
```bash
# Build container
docker build -t customer-propensity .

# Run container
docker run -p 8501:8501 customer-propensity
```

### **Access Dashboard**
Open your browser and go to `http://localhost:8501`

---

## 📁 Project Structure

```
customer-propensity/
├── 🌐 app.py                    # Streamlit web application
├── 🔧 pipeline.py              # Feature engineering pipeline
├── 🤖 model_evaluation.py      # Model evaluation script
├── 📦 propensity_model.pkl     # Trained XGBoost model
├── 📊 visitor_df_final.csv     # Processed features dataset
├── 📋 requirements.txt          # Python dependencies
├── 🐳 Dockerfile               # Container configuration
├── 📚 Readme.md                # Comprehensive documentation
├── 📖 PROJECT_SUMMARY.md       # Executive overview
├── 🚀 QUICK_START.md           # This guide
├── ⚡ run_evaluation.bat       # Windows execution script
├── ⚡ run_evaluation.sh        # Unix/Linux execution script
└── 📊 visualizations/          # Model evaluation plots
    ├── README.md               # Visualization guide
    ├── confusion_matrix.png    # Model accuracy
    ├── roc_curve.png          # ROC analysis
    ├── precision_recall.png   # Precision-recall
    ├── feature_importance.png # Feature ranking
    ├── threshold_analysis.png # Threshold optimization
    ├── class_distribution.png # Class analysis
    └── correlation_heatmap.png # Feature relationships
```

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **Model Loading Error**
```bash
# Ensure the model file exists
ls propensity_model.pkl

# Reinstall dependencies
pip install -r requirements.txt
```

#### **Data Loading Error**
```bash
# Check if data file exists
ls visitor_df_final.csv

# Verify file permissions
chmod 644 visitor_df_final.csv
```

#### **Package Installation Issues**
```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

#### **Streamlit Issues**
```bash
# Check Streamlit version
streamlit --version

# Clear Streamlit cache
streamlit cache clear
```

---

## 📚 Next Steps

### **Learn More**
1. **Read the full documentation** in `Readme.md`
2. **Explore the analysis notebook** `customer_propensity.ipynb`
3. **Review project summary** in `PROJECT_SUMMARY.md`

### **Customize & Extend**
1. **Modify features** in `pipeline.py`
2. **Update the dashboard** in `app.py`
3. **Add new models** to the evaluation script
4. **Deploy to cloud** using the Docker configuration

### **Business Applications**
1. **Marketing campaigns** targeting high-propensity customers
2. **Personalization** based on behavioral patterns
3. **Inventory management** using demand predictions
4. **Customer retention** strategies

---

## 🆘 Need Help?

### **Documentation**
- **README.md**: Complete setup and usage guide
- **PROJECT_SUMMARY.md**: Technical details and architecture
- **Jupyter Notebook**: Step-by-step development process

### **Execution Scripts**
- **Windows**: `run_evaluation.bat`
- **Unix/Linux**: `run_evaluation.sh`
- **Manual**: `python model_evaluation.py`

### **Support Resources**
- Check the troubleshooting section above
- Review error messages for specific guidance
- Verify all required files are present
- Ensure Python environment is properly configured

---

## 🎉 Success!

You now have a **production-ready customer propensity analysis system** running locally! 

**Key Benefits:**
- ✅ **98.20% accuracy** in purchase prediction
- ✅ **Real-time scoring** through interactive dashboard
- ✅ **Professional interface** with comprehensive metrics
- ✅ **Complete documentation** for maintenance and extension
- ✅ **Docker deployment** ready for production

**Ready to analyze customer behavior and predict purchase likelihood!** 🎯

---

*For detailed information, refer to the comprehensive documentation in `Readme.md` and `PROJECT_SUMMARY.md`*
