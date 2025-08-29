import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pipeline import create_features_for_visitor

# Page config
st.set_page_config(page_title="Customer Propensity Dashboard", layout="wide")

# Add custom CSS (theme-aware)
st.markdown("""
    <style>
    .main {padding: 1rem;}
    /* Card-like metric styling without forcing text color */
    .stMetric {padding: 1rem; border-radius: 5px;}
    @media (prefers-color-scheme: light) {
        .stMetric {background-color: #f0f2f6;}
    }
    @media (prefers-color-scheme: dark) {
        .stMetric {background-color: rgba(255,255,255,0.08);} /* subtle for dark */
    }
    .propensity-section {margin-bottom: 2rem; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px;}
    .high-propensity {background-color: #d4edda; border-color: #c3e6cb;}
    .medium-propensity {background-color: #fff3cd; border-color: #ffeaa7;}
    .low-propensity {background-color: #f8d7da; border-color: #f5c6cb;}
    .visitor-card {padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px; cursor: pointer;}
    @media (prefers-color-scheme: light) {
        .visitor-card {background-color: #ffffff;}
        .visitor-card:hover {background-color: #f8f9fa;}
    }
    @media (prefers-color-scheme: dark) {
        .visitor-card {background-color: rgba(255,255,255,0.06);} /* subtle dark card */
        .visitor-card:hover {background-color: rgba(255,255,255,0.12);}    
    }
    .selected-visitor {background-color: #007bff; color: white;}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title('🎯 Customer Purchase Propensity Dashboard')
st.markdown("""
    This dashboard analyzes customer purchase likelihood based on historical behavior data (up to 2015-09-18).
    Select a visitor from the sidebar to analyze their propensity to purchase.
""")

@st.cache_resource
def load_model():
    return joblib.load('propensity_model.pkl')

@st.cache_data
def load_data():
    events = pd.read_csv('data/events.csv')
    visitor_features = pd.read_csv('visitor_df_final.csv')
    return events, visitor_features

# Load data and model
model = load_model()
events_df, visitor_df = load_data()

# Sidebar with compact propensity sections
st.sidebar.header('📊 Customer Propensity Analysis')

# Initialize session state for selected visitor
if 'selected_visitor' not in st.session_state:
    st.session_state.selected_visitor = None

# Create a more compact sidebar layout
st.sidebar.markdown("---")

# 1. HIGH PROPENSITY USER SECTION
st.sidebar.markdown("### 🔴 High Propensity Users")
st.sidebar.markdown("*High purchase likelihood*")

# High propensity visitor IDs from the image
high_propensity_visitors = [
    {"id": 158090, "name": "Visitor 158090"},
    {"id": 1076270, "name": "Visitor 1076270"},
    {"id": 712443, "name": "Visitor 712443"},
    {"id": 599528, "name": "Visitor 599528"}
]

# Display high propensity visitors in a compact grid
cols = st.sidebar.columns(2)
for i, visitor in enumerate(high_propensity_visitors):
    col_idx = i % 2
    with cols[col_idx]:
        if st.button(
            f"{visitor['name']}",
            key=f"high_{visitor['id']}",
            help=f"Click to select {visitor['name']}",
            use_container_width=True
        ):
            st.session_state.selected_visitor = visitor['id']

st.sidebar.markdown("---")

# 2. MEDIUM PROPENSITY USER SECTION
st.sidebar.markdown("### 🟡 Medium Propensity Users")
st.sidebar.markdown("*Moderate purchase likelihood*")

# Medium propensity visitor IDs (False Negatives - Actual: 1, Predicted: 0)
medium_propensity_visitors = [
    {"id": 1199079, "name": "Visitor 1199079"},
    {"id": 1160955, "name": "Visitor 1160955"},
    {"id": 1240995, "name": "Visitor 1240995"},
    {"id": 526183, "name": "Visitor 526183"}
]

# Display medium propensity visitors in a compact grid
cols = st.sidebar.columns(2)
for i, visitor in enumerate(medium_propensity_visitors):
    col_idx = i % 2
    with cols[col_idx]:
        if st.button(
            f"{visitor['name']}",
            key=f"medium_{visitor['id']}",
            help=f"Click to select {visitor['name']}",
            use_container_width=True
        ):
            st.session_state.selected_visitor = visitor['id']

st.sidebar.markdown("---")

# 3. LOW PROPENSITY USER SECTION
st.sidebar.markdown("### 🟢 Low Propensity Users")
st.sidebar.markdown("*Low purchase likelihood*")

# Low propensity visitor IDs from the model evaluation
low_propensity_visitors = [
    {"id": 38299, "name": "Visitor 38299"},
    {"id": 921348, "name": "Visitor 921348"},
    {"id": 859383, "name": "Visitor 859383"},
    {"id": 699051, "name": "Visitor 699051"}
]

# Display low propensity visitors in a compact grid
cols = st.sidebar.columns(2)
for i, visitor in enumerate(low_propensity_visitors):
    col_idx = i % 2
    with cols[col_idx]:
        if st.button(
            f"{visitor['name']}",
            key=f"low_{visitor['id']}",
            help=f"Click to select {visitor['name']}",
            use_container_width=True
        ):
            st.session_state.selected_visitor = visitor['id']

# Analysis section
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Analysis Controls")

# Show selected visitor and clear option
if st.session_state.selected_visitor:
    st.sidebar.success(f"✅ Selected: Visitor {st.session_state.selected_visitor}")
    if st.sidebar.button("Clear selection", key="clear_selection"):
        st.session_state.selected_visitor = None
        if hasattr(st.session_state, 'analysis_results'):
            del st.session_state.analysis_results
else:
    st.sidebar.info("👆 Select a visitor from above sections")

# Predict button
if st.sidebar.button('🔍 Predict Propensity', key='predict_btn', type='primary', disabled=not st.session_state.selected_visitor):
    try:
        with st.spinner('Predicting propensity...'):
            features = create_features_for_visitor(st.session_state.selected_visitor, events_df)
            propensity_score = model.predict_proba(features)[:, 1][0]
            
            # Store results in session state
            st.session_state.analysis_results = {
                'visitor_id': st.session_state.selected_visitor,
                'propensity_score': propensity_score,
                'features': features
            }
        st.success(f"✅ Prediction completed for Visitor {st.session_state.selected_visitor}")
        
    except Exception as e:
        st.error(f"❌ Error predicting for visitor {st.session_state.selected_visitor}: {str(e)}")

# Main content area
if hasattr(st.session_state, 'analysis_results'):
    results = st.session_state.analysis_results
    
    # Main header with visitor info
    st.markdown(f"## 🔍 Customer Behavior Analysis - Visitor {results['visitor_id']}")

    # Prepare values for symmetric grid
    features = results['features']
    score = results['propensity_score']
    if score >= 0.7:
        color = "🟢"; status = "High Propensity"; bg_color = "#d4edda"
    elif score >= 0.4:
        color = "🟡"; status = "Medium Propensity"; bg_color = "#fff3cd"
    else:
        color = "🔴"; status = "Low Propensity"; bg_color = "#f8d7da"

    def card(icon, value, label, background="white"):
        bg_style = (
            "background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);"
            if background == "white" else f"background-color:{background};"
        )
        return f"""
        <div style=\"{bg_style}padding:1.25rem;border-radius:12px;border:1px solid #e6e6e6;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:140px;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0.5rem 0;\"> 
            <div style=\"font-size:1.6rem;margin-bottom:0.25rem;\">{icon}</div>
            <div style=\"font-size:1.6rem;font-weight:700;color:#222;\">{value}</div>
            <div style=\"font-size:0.95rem;color:#666;margin-top:0.25rem;\">{label}</div>
        </div>
        """

    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card("🎯", f"{score:.2%}", f"{color} {status}", bg_color), unsafe_allow_html=True)
    with c2:
        st.markdown(card("👁️", int(features['view_count'].values[0]), "Items Viewed"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("⏰", f"{int(features['recency_days'].values[0])} days", "Recency (days since last activity)"), unsafe_allow_html=True)

    # Row 2
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(card("🛒", int(features['addtocart_count'].values[0]), "Items in Cart"), unsafe_allow_html=True)
    with c5:
        st.markdown(card("🎯", int(features['unique_items_viewed'].values[0]), "Unique Items"), unsafe_allow_html=True)
    with c6:
        st.markdown(card("🔄", int(features['num_sessions'].values[0]), "Sessions"), unsafe_allow_html=True)

    # Row 3 (two cards with uniform spacing)
    # Simple heuristic CLV estimate
    aov = 50.0  # assumed average order value (USD)
    sessions_val = int(features['num_sessions'].values[0])
    cart_rate_val = float(features['add_to_cart_rate'].values[0])
    expected_purchases = sessions_val * cart_rate_val
    clv_est = max(0.0, score * expected_purchases * aov)
    c7, c8, c9 = st.columns(3)
    with c7:
        st.markdown(card("📈", f"{features['add_to_cart_rate'].values[0]:.2%}", "Cart Rate"), unsafe_allow_html=True)
    with c8:
        st.markdown(card("📊", f"{features['avg_events_per_session'].values[0]:.1f}", "Avg Events/Session"), unsafe_allow_html=True)
    with c9:
        st.markdown(card("💰", f"${clv_est:,.0f}", "CLV (est.)"), unsafe_allow_html=True)
    
    # Add a summary section below
    st.markdown("---")
    col_summary1, col_summary2 = st.columns([1, 1], gap="large")
    
    with col_summary1:
        st.markdown("### 📋 Quick Summary")
        st.markdown("*Based on historical data up to 2015-09-18*")
        
        if score >= 0.7:
            summary = "This customer showed strong purchase intent with high engagement levels."
            recommendation = "Focus on conversion optimization and premium offerings."
        elif score >= 0.4:
            summary = "This customer had moderate interest and may need nurturing."
            recommendation = "Implement targeted marketing campaigns and incentives."
        else:
            summary = "This customer showed low purchase likelihood during the analysis period."
            recommendation = "Focus on engagement and value proposition communication."
        
        st.info(f"**Analysis:** {summary}")
        st.success(f"**Recommendation:** {recommendation}")
    
    with col_summary2:
        st.markdown("### 🎯 Action Items")
        action_items = [
            "Monitor engagement patterns",
            "Track conversion funnel",
            "Analyze drop-off points",
            "Optimize user experience"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.markdown(f"<div style='margin: 8px 0'>{i}. {item}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Dashboard created for customer behavior analysis*")
st.markdown("<div style='text-align: center; color: #8a8a8a; margin-top: 4px;'>Made by Himanshu Raj</div>", unsafe_allow_html=True)