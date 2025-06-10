"""Exploratory Data Analysis page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.visualiser import Visualiser


def _group_features(df):
    """Group features into logical categories for better organisation.
    
    Args:
        df (pd.DataFrame): Input dataframe containing features to group.
        
    Returns:
        dict: Dictionary mapping group names to lists of feature names.
              Keys are group names (str), values are lists of column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Define feature groups based on NSL-KDD dataset structure
    feature_groups = {
        "Basic Connection Features": [
            col for col in numeric_cols 
            if col in ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent']
        ],
        "Content Features": [
            col for col in numeric_cols 
            if col in ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
                      'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
                      'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login']
        ],
        "Traffic Features": [
            col for col in numeric_cols 
            if col in ['count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
                      'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']
        ],
        "Host-based Features": [
            col for col in numeric_cols 
            if col in ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
        ],
        "Categorical Features": categorical_cols,
        "Other Numerical": [
            col for col in numeric_cols 
            if col not in [item for sublist in [
                ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent'],
                ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
                 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
                 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login'],
                ['count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
                 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'],
                ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
            ] for item in sublist] and 'labels' not in col
        ]
    }
    
    # Remove empty groups
    feature_groups = {k: v for k, v in feature_groups.items() if v}
    
    return feature_groups


def _create_grouped_feature_selector(df, title="Select Features", key_prefix="selector"):
    """Create an organised feature selector with groups and select all options.
    
    Args:
        df (pd.DataFrame): Input dataframe containing features.
        title (str, optional): Title for the feature selector. Defaults to "Select Features".
        key_prefix (str, optional): Prefix for Streamlit widget keys. Defaults to "selector".
        
    Returns:
        list: List of selected feature names (column names from the dataframe).
    """
    feature_groups = _group_features(df)
    
    st.markdown(f"#### {title}")
    
    selected_features = []
    
    # Select All option
    col1, col2 = st.columns([1, 3])
    with col1:
        select_all = st.checkbox("Select All", key=f"{key_prefix}_select_all")
    
    if select_all:
        # If select all is checked, return all features
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        return all_features
    
    # Create expandable sections for each group
    for group_name, group_features in feature_groups.items():
        if group_features:  # Only show groups with features
            with st.expander(f"📁 {group_name} ({len(group_features)} features)", expanded=False):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    select_group = st.checkbox(
                        f"Select All {group_name}", 
                        key=f"{key_prefix}_{group_name}_all"
                    )
                
                if select_group:
                    selected_features.extend(group_features)
                    with col2:
                        st.info(f"All {len(group_features)} features selected")
                else:
                    with col2:
                        group_selected = st.multiselect(
                            f"Choose from {group_name}:",
                            group_features,
                            key=f"{key_prefix}_{group_name}"
                        )
                        selected_features.extend(group_selected)
    
    return selected_features


def show_eda_page():
    """Display the Exploratory Data Analysis page.
    
    This function creates the main EDA interface with multiple tabs for different
    types of analysis including target analysis, feature analysis, correlations,
    feature statistics, and clustering.
    
    Returns:
        None: Renders the EDA interface directly to Streamlit.
    """
    st.title("📊 Exploratory Data Analysis")

    if not st.session_state.get('data_loaded', False):
        st.warning("Please upload data first!")
        return

    df = st.session_state.data
    processor = st.session_state.processor
    visualiser = Visualiser()

    # Create processed dataframe with labels
    df_processed = processor.create_target_labels(df)

    # Global EDA settings in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎛️ EDA Settings")
        
        # Sample size for performance
        sample_size = st.slider(
            "Sample Size", 
            min_value=1000, 
            max_value=min(50000, len(df)), 
            value=min(10000, len(df)),
            help="Reduce for better performance with large datasets"
        )
        
        # Apply sampling
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            df_processed_sample = processor.create_target_labels(df_sample)
            st.info(f"Using {sample_size:,} samples for visualisation")
        else:
            df_sample = df
            df_processed_sample = df_processed

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Target Analysis", "Feature Analysis", "Correlations", "Feature Statistics", "Clustering"]
    )

    with tab1:
        _show_target_analysis(df_processed_sample, processor, visualiser)

    with tab2:
        _show_feature_analysis(df_sample, visualiser)

    with tab3:
        _show_correlation_analysis(df_sample, visualiser)

    with tab4:
        _show_feature_statistics(df_sample, df_processed_sample)

    with tab5:
        _show_clustering_analysis(df_processed_sample, processor, visualiser)


def _show_target_analysis(df_processed, processor, visualiser):
    """Show target variable analysis.
    
    Args:
        df_processed (pd.DataFrame): Processed dataframe with target labels.
        processor: Data processor instance for accessing attack mappings.
        visualiser: Visualiser instance for creating plots.
        
    Returns:
        None: Renders analysis directly to Streamlit.
    """
    st.markdown("### Target Variable Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Binary classification distribution
        fig = visualiser.plot_target_distribution(
            df_processed, "labels2", "Binary Classification Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Multi-class distribution
        fig = visualiser.plot_target_distribution(
            df_processed, "labels5", "Multi-class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Attack type mapping
    st.markdown("### Attack Type Mapping")
    attack_df = pd.DataFrame(
        list(processor.attack_mapping.items()), columns=["Attack", "Category"]
    )
    attack_df = attack_df[attack_df["Attack"] != "normal"].sort_values("Category")

    col1, col2, col3, col4 = st.columns(4)
    categories = attack_df["Category"].unique()
    for i, cat in enumerate(categories):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"**{cat}**")
            attacks = attack_df[attack_df["Category"] == cat]["Attack"].tolist()
            
            # Show first 5 attacks
            for attack in attacks[:5]:
                st.write(f"• {attack}")
            
            # If more than 5 attacks, show expandable section
            if len(attacks) > 5:
                with st.expander(f"Show {len(attacks) - 5} more {cat} attacks"):
                    for attack in attacks[5:]:
                        st.write(f"• {attack}")


def _show_feature_analysis(df, visualiser):
    """Show feature analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        visualiser: Visualiser instance for creating plots.
        
    Returns:
        None: Renders analysis directly to Streamlit.
    """
    st.markdown("### 🔍 Interactive Feature Analysis")

    # User controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical feature selection
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        selected_categorical = st.selectbox(
            "Select Categorical Feature",
            categorical_cols,
            index=0 if "service" in categorical_cols else 0,
            help="Choose which categorical feature to analyse"
        )
        
    with col2:
        # Top N for categorical features
        top_n_categorical = st.slider(
            "Top N Categories",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of top categories to display"
        )

    # Display selected categorical feature - Always use horizontal bar
    if selected_categorical:
        st.markdown(f"#### {selected_categorical.replace('_', ' ').title()} Distribution")
        
        counts = df[selected_categorical].value_counts().head(top_n_categorical)
        # Create horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            x=counts.values,
            y=counts.index,
            orientation='h',
            marker_color='lightblue'
        )])
        fig.update_layout(
            title=f"{selected_categorical.replace('_', ' ').title()} Distribution",
            xaxis_title="Count",
            yaxis_title=selected_categorical.replace('_', ' ').title(),
            height=max(400, len(counts) * 25),  # Dynamic height based on number of categories
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show value counts table
        with st.expander(f"📊 {selected_categorical.title()} Value Counts"):
            value_counts = df[selected_categorical].value_counts()
            st.dataframe(value_counts.to_frame("Count"), use_container_width=True)

    # Numerical features analysis
    st.markdown("---")
    st.markdown("#### 📈 Numerical Features Analysis")
    
    # Use grouped feature selector
    selected_numeric = _create_grouped_feature_selector(
        df, 
        title="Select Numerical Features", 
        key_prefix="feature_analysis_numeric"
    )
    
    # Remove non-numeric features that might have been selected
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_numeric = [col for col in selected_numeric if col in numeric_cols and 'labels' not in col]
    
    if selected_numeric:
        st.info(f"Selected {len(selected_numeric)} features")
        _show_distribution_plots(df, selected_numeric, visualiser)
    else:
        st.info("Please select numerical features to analyse.")


def _show_correlation_analysis(df, visualiser):
    """Show correlation analysis using heatmaps.
    
    Args:
        df (pd.DataFrame): Input dataframe for correlation analysis.
        visualiser: Visualiser instance for creating plots.
    """
    st.markdown("### 🔗 Interactive Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        st.warning("Not enough numerical features for correlation analysis.")
        return
    
    # Use grouped feature selector
    selected_features = _create_grouped_feature_selector(
        df, 
        title="Select Features for Correlation Analysis", 
        key_prefix="correlation_analysis"
    )
    
    # Filter to only numeric features
    numeric_features = [col for col in selected_features if col in numeric_df.columns and 'labels' not in col]
    
    if not numeric_features or len(numeric_features) < 2:
        st.warning("Please select at least 2 numerical features for correlation analysis.")
        return
    
    # Correlation method selection
    corr_method = st.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"],
        help="Method for calculating correlations"
    )
    
    st.info(f"Selected {len(numeric_features)} numerical features for correlation analysis")
    
    # Generate correlation heatmap
    corr_data = numeric_df[numeric_features]
    with st.spinner("Generating correlation heatmap..."):
        fig = visualiser.plot_correlation_heatmap(corr_data)
        st.pyplot(fig)
    
    # High correlations table
    st.markdown("#### 🎯 Highly Correlated Feature Pairs")
    
    col1, col2 = st.columns(2)
    with col1:
        correlation_threshold = st.slider(
            "Correlation Threshold", 
            0.5, 0.99, 0.8, 0.05,
            help="Threshold for displaying highly correlated features"
        )
    with col2:
        show_negative = st.checkbox(
            "Include Negative Correlations",
            value=True,
            help="Show both positive and negative correlations"
        )
    
    # Calculate correlations and find high correlations
    corr_matrix = corr_data.corr(method=corr_method)
    high_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            condition = abs(corr_val) > correlation_threshold if show_negative else corr_val > correlation_threshold
            
            if condition:
                high_corr.append({
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": corr_val,
                    "Abs Correlation": abs(corr_val),
                    "Type": "Positive" if corr_val > 0 else "Negative"
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values("Abs Correlation", ascending=False)
        
        # Colour code the dataframe
        def colour_correlation(val):
            if val > 0.8:
                return 'background-color: #ffcccc'
            elif val < -0.8:
                return 'background-color: #ccccff'
            elif val > 0.6:
                return 'background-color: #ffffcc'
            return ''
        
        styled_df = high_corr_df.style.applymap(colour_correlation, subset=['Correlation'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Download option
        csv = high_corr_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Correlation Data",
            data=csv,
            file_name="correlation_analysis.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No highly correlated features found (threshold: {correlation_threshold})")


def _show_clustering_analysis(df_processed, processor, visualiser):
    """Show clustering analysis with DBSCAN and dimensionality reduction.
    
    Args:
        df_processed (pd.DataFrame): Processed dataframe with target labels.
        processor: Data processor instance for accessing label information.
        visualiser: Visualiser instance for creating plots.
        
    Returns:
        None: Renders analysis directly to Streamlit.
    """
    st.markdown("### 🎯 Clustering Analysis")

    # Prepare data
    X = df_processed.drop(columns=["labels", "labels2", "labels5"])
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numerical features available for clustering analysis.")
        return
        
    X_numeric = X[numeric_cols]

    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sample size for performance
        sample_size = st.slider(
            "Sample Size",
            1000, min(10000, len(X_numeric)), 3000,
            help="Number of samples to use for clustering (for performance)"
        )
        
    with col2:
        # Dimensionality reduction method
        dim_method = st.selectbox(
            "Dimensionality Reduction",
            ["UMAP", "t-SNE"],
            help="Method for reducing dimensions before visualisation"
        )
    
    with col3:
        # Clustering algorithm
        cluster_method = st.selectbox(
            "Clustering Algorithm",
            ["DBSCAN", "KMeans"],
            help="Clustering algorithm to use"
        )

    # Algorithm-specific parameters
    st.markdown("#### Algorithm Parameters")
    
    if cluster_method == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5, 0.1)
        with col2:
            min_samples = st.slider("Min Samples", 3, 20, 5)
    else:  # KMeans
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 15, 5)
        with col2:
            random_state = st.slider("Random State", 1, 100, 42)

    # Feature selection for clustering
    st.markdown("#### Feature Selection")
    selected_features = _create_grouped_feature_selector(
        df_processed, 
        title="Select Features for Clustering", 
        key_prefix="clustering_analysis"
    )
    
    # Filter to numeric features only
    numeric_features = [col for col in selected_features if col in numeric_cols and 'labels' not in col]
    
    if not numeric_features:
        st.warning("Please select numerical features for clustering.")
        return
    
    if len(numeric_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
        return

    # Run clustering analysis
    if st.button("🚀 Run Clustering Analysis", type="primary"):
        with st.spinner("Running clustering analysis..."):
            try:
                # Sample data if too large
                if len(X_numeric) > sample_size:
                    sample_idx = np.random.choice(len(X_numeric), sample_size, replace=False)
                    X_sample = X_numeric[numeric_features].iloc[sample_idx]
                    y_sample = df_processed["labels5"].iloc[sample_idx]
                else:
                    X_sample = X_numeric[numeric_features]
                    y_sample = df_processed["labels5"]
                
                # Standardize features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_sample)
                
                # Apply clustering
                if cluster_method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = clusterer.fit_predict(X_scaled)
                else:  # KMeans
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    cluster_labels = clusterer.fit_predict(X_scaled)
                
                # Apply dimensionality reduction for visualisation
                if dim_method == "UMAP":
                    try:
                        import umap.umap_ as umap
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        X_reduced = reducer.fit_transform(X_scaled)
                    except ImportError:
                        st.error("UMAP not available. Install with: pip install umap-learn")
                        return
                else:  # t-SNE
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
                    X_reduced = reducer.fit_transform(X_scaled)
                
                # Create visualisations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Clustering results
                    fig = go.Figure()
                    
                    unique_clusters = np.unique(cluster_labels)
                    colors = px.colors.qualitative.Set3
                    
                    for i, cluster in enumerate(unique_clusters):
                        mask = cluster_labels == cluster
                        cluster_name = f"Cluster {cluster}" if cluster != -1 else "Noise"
                        
                        fig.add_trace(go.Scatter(
                            x=X_reduced[mask, 0],
                            y=X_reduced[mask, 1],
                            mode='markers',
                            name=cluster_name,
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.7
                            )
                        ))
                    
                    fig.update_layout(
                        title=f"{cluster_method} Clustering Results ({dim_method} visualisation)",
                        xaxis_title=f"{dim_method} Component 1",
                        yaxis_title=f"{dim_method} Component 2",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # True labels for comparison
                    fig = go.Figure()
                    
                    target_names = processor.get_label_names("labels5")
                    unique_labels = np.unique(y_sample)
                    
                    for i, label in enumerate(unique_labels):
                        mask = y_sample == label
                        label_name = target_names[label] if label < len(target_names) else f"Class {label}"
                        
                        fig.add_trace(go.Scatter(
                            x=X_reduced[mask, 0],
                            y=X_reduced[mask, 1],
                            mode='markers',
                            name=label_name,
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.7
                            )
                        ))
                    
                    fig.update_layout(
                        title=f"True Labels ({dim_method} visualisation)",
                        xaxis_title=f"{dim_method} Component 1",
                        yaxis_title=f"{dim_method} Component 2",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clustering metrics
                st.markdown("### 📊 Clustering Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))
                    st.metric("Clusters Found", n_clusters_found)
                
                with col2:
                    n_noise = np.sum(cluster_labels == -1)
                    noise_pct = (n_noise / len(cluster_labels)) * 100
                    st.metric("Noise Points", f"{n_noise} ({noise_pct:.1f}%)")
                
                with col3:
                    largest_cluster = np.max(np.bincount(cluster_labels[cluster_labels != -1]))
                    st.metric("Largest Cluster", largest_cluster)
                
                # Silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    from sklearn.metrics import silhouette_score
                    try:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        st.metric("Silhouette Score", f"{sil_score:.3f}")
                    except:
                        st.info("Silhouette score could not be calculated")
                
                # Feature importance (for KMeans)
                if cluster_method == "KMeans" and hasattr(clusterer, 'cluster_centers_'):
                    st.markdown("### 🎯 Feature Importance in Clustering")
                    
                    # Calculate feature importance based on cluster centre variations
                    centers = clusterer.cluster_centers_
                    feature_variance = np.var(centers, axis=0)
                    
                    importance_df = pd.DataFrame({
                        'Feature': numeric_features,
                        'Importance': feature_variance
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Features by Clustering Importance"
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Clustering analysis completed using {len(numeric_features)} features!")
                
            except Exception as e:
                st.error(f"Clustering analysis failed: {str(e)}")
                st.info("Try adjusting the parameters or selecting different features.")

    st.info("""
    **Note**: 
    - **DBSCAN** automatically finds the number of clusters and identifies noise points
    - **KMeans** requires you to specify the number of clusters
    - **UMAP** generally preserves more global structure than t-SNE
    - **t-SNE** is good for visualising local neighbourhoods
    """)


def _show_feature_statistics(df, df_processed):
    """Show detailed feature statistics.
    
    Args:
        df (pd.DataFrame): Original dataframe for analysis.
        df_processed (pd.DataFrame): Processed dataframe with target labels.
        
    Returns:
        None: Renders statistics directly to Streamlit.
    """
    st.markdown("### 📈 Interactive Feature Statistics")
    
    # Feature type selection
    col1, col2 = st.columns(2)
    
    with col1:
        feature_type = st.selectbox(
            "Feature Type",
            ["All", "Numerical", "Categorical", "Binary"],
            help="Filter features by type"
        )
    
    with col2:
        stat_type = st.selectbox(
            "Statistics Type",
            ["Descriptive", "Distribution", "Missing Values", "Outliers"],
            help="Type of statistical analysis"
        )
    
    # Get features based on selection
    if feature_type == "Numerical":
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    elif feature_type == "Categorical":
        features = df.select_dtypes(include=['object']).columns.tolist()
    elif feature_type == "Binary":
        features = [col for col in df.columns if df[col].nunique() == 2]
    else:  # All
        features = df.columns.tolist()
    
    if not features:
        st.warning(f"No {feature_type.lower()} features found.")
        return
    
    # Use grouped feature selector
    if feature_type == "All":
        selected_features = _create_grouped_feature_selector(
            df, 
            title="Select Features to Analyze", 
            key_prefix=f"feature_stats_{feature_type}"
        )
        # Filter based on feature type if needed
        selected_features = [col for col in selected_features if col in features]
    else:
        # For specific types, use simple multiselect
        selected_features = st.multiselect(
            "Select Features to Analyze",
            features,
            default=features[:10] if len(features) > 10 else features,
            help="Choose specific features for analysis"
        )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Display statistics based on type
    if stat_type == "Descriptive":
        _show_descriptive_stats(df, selected_features)
    elif stat_type == "Distribution":
        _show_distribution_stats(df, selected_features)
    elif stat_type == "Missing Values":
        _show_missing_values_analysis(df, selected_features)
    elif stat_type == "Outliers":
        _show_outlier_analysis(df, selected_features)


def _show_descriptive_stats(df, features):
    """Show descriptive statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        features (list): List of feature names to analyse.
        
    Returns:
        None: Renders statistics directly to Streamlit.
    """
    st.markdown("#### 📊 Descriptive Statistics")
    
    numeric_features = [f for f in features if df[f].dtype in [np.number]]
    categorical_features = [f for f in features if df[f].dtype == 'object']
    
    if numeric_features:
        st.markdown("**Numerical Features:**")
        stats_df = df[numeric_features].describe()
        st.dataframe(stats_df.round(4), use_container_width=True)
        
        # Additional statistics
        additional_stats = pd.DataFrame({
            'Skewness': df[numeric_features].skew(),
            'Kurtosis': df[numeric_features].kurtosis(),
            'Variance': df[numeric_features].var()
        }).round(4)
        
        with st.expander("Advanced Statistical Measures"):
            st.dataframe(additional_stats, use_container_width=True)
    
    if categorical_features:
        st.markdown("**Categorical Features:**")
        cat_stats = []
        for feature in categorical_features:
            stats = {
                'Feature': feature,
                'Unique Values': df[feature].nunique(),
                'Most Frequent': df[feature].mode().iloc[0] if not df[feature].mode().empty else 'N/A',
                'Most Frequent Count': df[feature].value_counts().iloc[0] if len(df[feature].value_counts()) > 0 else 0,
                'Missing Values': df[feature].isnull().sum()
            }
            cat_stats.append(stats)
        
        cat_df = pd.DataFrame(cat_stats)
        st.dataframe(cat_df, use_container_width=True)


def _show_distribution_stats(df, features):
    """Show distribution analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        features (list): List of feature names to analyse.
        
    Returns:
        None: Renders distribution plots directly to Streamlit.
    """
    st.markdown("#### 📈 Distribution Analysis")
    
    numeric_features = [f for f in features if df[f].dtype in [np.number]]
    
    if not numeric_features:
        st.warning("No numerical features selected for distribution analysis.")
        return
    
    # Distribution plots
    for feature in numeric_features[:6]:  # Limit to first 6 features
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=feature, title=f"Box Plot of {feature}")
            st.plotly_chart(fig, use_container_width=True)


def _show_missing_values_analysis(df, features):
    """Show missing values analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        features (list): List of feature names to analyse.
        
    Returns:
        None: Renders missing values analysis directly to Streamlit.
    """
    st.markdown("#### 🕳️ Missing Values Analysis")
    
    missing_data = []
    for feature in features:
        missing_count = df[feature].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_data.append({
            'Feature': feature,
            'Missing Count': missing_count,
            'Missing Percentage': missing_pct,
            'Data Type': str(df[feature].dtype)
        })
    
    missing_df = pd.DataFrame(missing_data)
    missing_df = missing_df.sort_values('Missing Count', ascending=False)
    
    # Colour code based on missing percentage
    def colour_missing(val):
        if val > 50:
            return 'background-color: #ffcccc'  # Red for high missing
        elif val > 20:
            return 'background-color: #ffffcc'  # Yellow for medium missing
        elif val > 0:
            return 'background-color: #ccffcc'  # Green for low missing
        return ''
    
    styled_df = missing_df.style.applymap(colour_missing, subset=['Missing Percentage'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Missing values heatmap
    if len(features) > 1:
        st.markdown("**Missing Values Heatmap:**")
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features].isnull(), cbar=True, ax=ax, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        st.pyplot(fig)


def _show_outlier_analysis(df, features):
    """Show outlier analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        features (list): List of feature names to analyse.
        
    Returns:
        None: Renders outlier analysis directly to Streamlit.
    """
    st.markdown("#### 🎯 Outlier Analysis")
    
    numeric_features = [f for f in features if df[f].dtype in [np.number]]
    
    if not numeric_features:
        st.warning("No numerical features selected for outlier analysis.")
        return
    
    # Outlier detection method
    method = st.selectbox(
        "Outlier Detection Method",
        ["IQR Method", "Z-Score", "Modified Z-Score"],
        help="Method for detecting outliers"
    )
    
    outlier_data = []
    
    for feature in numeric_features:
        if method == "IQR Method":
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            
        elif method == "Z-Score":
            z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
            outliers = df[z_scores > 3]
            
        elif method == "Modified Z-Score":
            median = df[feature].median()
            mad = np.median(np.abs(df[feature] - median))
            modified_z_scores = 0.6745 * (df[feature] - median) / mad
            outliers = df[np.abs(modified_z_scores) > 3.5]
        
        outlier_data.append({
            'Feature': feature,
            'Total Outliers': len(outliers),
            'Outlier Percentage': (len(outliers) / len(df)) * 100,
            'Min Value': df[feature].min(),
            'Max Value': df[feature].max()
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    outlier_df = outlier_df.sort_values('Outlier Percentage', ascending=False)
    
    st.dataframe(outlier_df.round(4), use_container_width=True)
    
    # Outlier visualisation
    if len(numeric_features) > 0:
        selected_feature = st.selectbox("Select Feature for Outlier Visualisation", numeric_features)
        
        fig = px.box(df, y=selected_feature, title=f"Outliers in {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)


def _show_distribution_plots(df, features, visualiser):
    """Show distribution plots for numerical features with independent scales.
    
    Args:
        df (pd.DataFrame): Input dataframe for analysis.
        features (list): List of feature names to analyse.
        visualiser: Visualiser instance for creating plots (not used in current implementation).
        
    Returns:
        None: Renders distribution plots directly to Streamlit.
    """
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j, feature in enumerate(features[i:i+3]):
            with cols[j]:
                fig = px.histogram(
                    df, 
                    x=feature, 
                    marginal="box", 
                    title=f"Distribution of {feature}"
                )
                # Ensure independent scaling for each plot
                fig.update_layout(
                    autosize=True,
                    xaxis=dict(autorange=True),
                    yaxis=dict(autorange=True),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)