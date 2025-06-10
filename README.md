# Network Intrusion Detection System

A modern machine learning-based intrusion detection system built with Python and Streamlit. Detects network intrusions using the NSL-KDD dataset with support for both binary and multi-class classification.

## 🚀 Features

### Core Functionality
- 🛡️ **Binary Classification**: Normal vs Attack traffic detection
- 🎯 **Multi-class Classification**: Specific attack type identification (DoS, Probe, R2L, U2R)
- 🔄 **Two-Phase Detection**: Binary → Multi-class classification pipeline
- 🤖 **6+ ML Algorithms**: Random Forest, Decision Tree, Logistic Regression (balanced variants included)

### Advanced Capabilities
- 🔧 **Customisable Feature Engineering**: Automated feature selection, scaling, and preprocessing
- 📊 **Interactive EDA**: Comprehensive exploratory data analysis with clustering and dimensionality reduction
- 🔮 **Real-time Predictions**: Single prediction interface with confidence scores
- 🎨 **Rich Visualisations**: Correlation heatmaps, feature importance, confusion matrices
- 📱 **Auto-loading System**: Automatic dataset detection and preprocessing

### Technical Excellence
- 🏗️ **Modular Architecture**: Component-based Streamlit structure
- 🎯 **Session Management**: Persistent state across navigation
- 📈 **Performance Optimised**: Sampling options for large datasets
- 🔍 **Google Docstrings**: Comprehensive documentation throughout

## 🛠️ Installation & Setup

### Prerequisites
- Python ≥3.12
- NSL-KDD dataset files

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd network-intrusion-detection
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Download NSL-KDD dataset** and place in `data/` directory:
   - `data/KDDTrain+.csv` - Training dataset
   - `data/KDDTest+.csv` - Test dataset
   - Download from: https://www.unb.ca/cic/datasets/nsl.html

4. **Run the application**:
   ```bash
   streamlit run app/app.py
   ```

5. **Access the interface**: Open http://localhost:8501 in your browser

## 📁 Project Structure

```
├── app/
│   ├── app.py                    # Main Streamlit application router
│   └── components/
│       ├── home.py               # Auto-loading dashboard  
│       ├── data_upload.py        # Dataset management
│       ├── eda.py                # Interactive data exploration
│       ├── model_training.py     # Model training interface
│       ├── predictions.py        # Real-time predictions
│       └── about.py              # Documentation
├── src/
│   ├── data_processor.py         # NSL-KDD preprocessing & attack mapping
│   ├── feature_engineer.py      # Feature selection & preprocessing pipeline
│   ├── models.py                 # ML models & evaluation metrics
│   ├── utils.py                  # Utility functions & configurations
│   └── visualiser.py            # Visualisation utilities
├── data/                         # NSL-KDD dataset files
├── pyproject.toml               # Project configuration & dependencies
└── uv.lock                      # Dependency lock file
```

## 🎯 Usage Guide

### 1. Data Loading
The system automatically detects NSL-KDD files in the `data/` directory and loads them on startup.

### 2. Exploratory Data Analysis
- **Target Analysis**: Distribution of attack types and categories
- **Feature Analysis**: Interactive numerical and categorical feature exploration
- **Correlation Analysis**: Heatmaps and highly correlated feature identification
- **Clustering**: DBSCAN/KMeans with UMAP/t-SNE visualisation

### 3. Model Training
- Choose from 6+ ML algorithms (including balanced variants)
- Configure feature engineering pipeline with custom settings
- Select binary or two-phase classification approach
- View comprehensive performance metrics and feature importance

### 4. Predictions
- Input network connection features through dynamic form
- Get real-time predictions with confidence scores
- Support for both binary and multi-class prediction modes

## 🔧 Configuration

### Feature Engineering Options
- **Feature Selection**: Remove constant, duplicate, and correlated features
- **Scaling Methods**: Standard, MinMax, Robust scaling
- **Categorical Encoding**: Ordinal, OneHot, Target encoding
- **Rare Label Handling**: Configurable frequency thresholds

### Model Options
- **Algorithms**: Random Forest, Decision Tree, Logistic Regression
- **Balanced Variants**: Built-in class balancing for imbalanced datasets
- **Classification Types**: Multi-class (5 classes) or Two-phase detection
- **Evaluation**: Comprehensive metrics including confusion matrices

## 📊 Dataset Information

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset:
- **Training samples**: ~125,973 records
- **Test samples**: ~22,544 records  
- **Features**: 41 features per connection record
- **Attack categories**: DoS, Probe, R2L, U2R + Normal
- **Attack types**: 39 specific attack types mapped to 4 main categories

## 🚀 Version 2.0 Improvements

### New Features
- **Auto-loading System**: Automatic dataset detection and processing
- **Component Architecture**: Modular Streamlit pages for better maintainability
- **Enhanced EDA**: Interactive analysis with clustering and dimensionality reduction
- **Streamlined Predictions**: Simplified single prediction interface
- **Session Management**: Persistent state across navigation

### Technical Improvements
- **3-column Plot Layout**: Optimised visualisation display (3 plots per row)
- **Grouped Feature Selection**: Organised feature selectors for better UX
- **Fixed Prediction Pipeline**: Resolved feature mismatch issues
- **Performance Optimisation**: Sampling options for large datasets
- **Better Error Handling**: Comprehensive error messages and user feedback

### Code Quality
- **British Spelling**: Consistent spelling conventions throughout
- **Google Docstrings**: Professional documentation format
- **Modern Python**: Python 3.12+ features and type hints
- **Clean Architecture**: Single responsibility principle

## 🤝 Contributing

This project follows conventional commit practices and maintains high code quality standards. See CLAUDE.md for detailed development guidelines.

## 📄 License

MIT License - see LICENSE file for details.