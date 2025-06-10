"""About page for the application."""

import streamlit as st


def show_about_page():
    """Display the about page.
    
    Shows comprehensive information about the Network Intrusion Detection System,
    including features, technical stack, dataset information, and usage instructions.
    """
    st.title("ℹ️ About")

    st.markdown("""
    ### Network Intrusion Detection System
    
    This application implements a machine learning-based Intrusion Detection System (IDS) 
    using the NSL-KDD dataset. It provides both binary and multi-class classification 
    capabilities to detect and categorize network intrusions.
    
    #### 🎯 Features
    - **Auto Data Loading**: Automatically loads NSL-KDD dataset if available
    - **Customizable Feature Engineering**: Advanced feature selection and transformation options
    - **Multiple ML Models**: Support for various algorithms with balanced options
    - **Interactive Visualisations**: Real-time data exploration and analysis
    - **Two-phase Detection**: Binary classification followed by attack type identification
    - **Modular Architecture**: Clean, maintainable code structure
    
    #### 🚨 Attack Categories
    1. **DoS (Denial of Service)**: Attacks that make resources unavailable to legitimate users
    2. **Probe**: Surveillance and scanning attacks to gather information
    3. **R2L (Remote to Local)**: Unauthorized access attempts from remote machines
    4. **U2R (User to Root)**: Unauthorized attempts to gain root/admin privileges
    
    #### 🛠️ Technical Stack
    - **Frontend**: Streamlit
    - **ML Framework**: scikit-learn
    - **Feature Engineering**: feature-engine
    - **Visualisation**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    - **Package Manager**: uv
    
    #### 📊 Performance Capabilities
    - **Binary Classification**: Up to 99.9% accuracy
    - **Multi-class Classification**: Up to 99.7% accuracy
    - **Real-time Prediction**: Fast inference on new data
    - **Batch Processing**: Handle large datasets efficiently
    
    #### 📚 Dataset Information
    The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset, 
    addressing some of the inherent problems mentioned in the literature:
    
    - **Training Set**: 125,973 records
    - **Test Set**: 22,544 records  
    - **Features**: 41 features (38 numerical, 3 categorical)
    - **Classes**: 5 classes (Normal + 4 attack types)
    - **Attack Types**: 39 different specific attacks
    
    #### 🔧 Model Features
    
    **Available Algorithms:**
    - Random Forest (balanced and standard)
    - Decision Tree (balanced and standard)
    - Logistic Regression (balanced and standard)
    
    **Feature Engineering Options:**
    - Constant feature removal
    - Duplicate feature removal
    - Correlation-based feature selection
    - Rare label encoding
    - Multiple scaling methods
    - Categorical encoding strategies
    
    **Detection Strategies:**
    - **Multi-class**: Direct classification into 5 categories
    - **Two-phase**: Binary detection followed by attack classification
    
    #### 🚀 Getting Started
    
    1. **Data**: Place NSL-KDD files in `data/` directory for auto-loading
    2. **Explore**: Use the EDA Dashboard to understand your data
    3. **Configure**: Customize feature engineering in Model Training
    4. **Train**: Select and train your preferred model
    5. **Predict**: Test on single connections or batch data
    
    #### 📁 Project Structure
    ```
    app/
    ├── app.py              # Main application entry point
    └── pages/              # Modular page components
        ├── home.py         # Home page with auto-loading
        ├── data_upload.py  # Data management
        ├── eda.py          # Exploratory data analysis
        ├── model_training.py # Training with custom FE
        ├── predictions.py  # Improved predictions
        └── about.py        # This page
    
    src/
    ├── data_processor.py   # Data loading and preprocessing
    ├── feature_engineer.py # Feature engineering pipeline
    ├── models.py           # ML model implementations
    ├── visualiser.py       # Visualisation utilities
    └── utils.py            # Helper functions
    ```
    
    #### 🔒 Security Considerations
    
    This is a research and educational tool. For production deployment:
    - Implement proper authentication and authorization
    - Add input validation and sanitization
    - Use secure data handling practices
    - Monitor for adversarial attacks
    - Regular model retraining with new data
    
    #### 📖 References
    
    - **NSL-KDD Dataset**: [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
    - **Original KDD Cup 1999**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data)
    - **Feature Engineering**: [feature-engine Documentation](https://feature-engine.readthedocs.io/)
    
    ---
    
    **Version**: 2.0.0  
    **License**: MIT  
    **GitHub**: [network-intrusion-detection](https://github.com/yourusername/network-intrusion-detection)
    
    ---
    
    <div style='text-align: centre; margin-top: 2rem;'>
        <p style='color: #888;'>Made with ❤️ using Streamlit and Python</p>
        <p style='color: #888;'>For educational and research purposes</p>
    </div>
    """, unsafe_allow_html=True)