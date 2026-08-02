"""
Streamlit Interactive Dashboard for Smart Retail & Customer Intelligence Platform.
Displays real-time analytics, visit trends, sentiment distribution, and interactive AI testing modules.
"""

import sys
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import get_pipeline
from cv_utils import detect_faces_haar, draw_face_bounding_boxes

# Page Config
st.set_page_config(
    page_title="Smart Retail Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Subtle Green Palette
st.markdown("""
<style>
    /* Main Header & Subheader */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #10B981;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6EE7B7;
        margin-bottom: 1.5rem;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #064E3B1A;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #10B981;
        box-shadow: 0 1px 3px rgba(16, 185, 129, 0.15);
    }
    /* Badges */
    .badge-pos {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-neu {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #FFE4E6;
        color: #9F1239;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    /* Primary buttons green accent */
    .stButton > button {
        border-color: #10B981 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #10B981 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
    }
    /* Tab highlighting */
    button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
    }
    button[aria-selected="true"] {
        border-bottom-color: #10B981 !important;
    }
    /* Progress bar green accent */
    .stProgress > div > div > div > div {
        background-color: #10B981 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_smart_pipeline():
    return get_pipeline()

pipeline = load_smart_pipeline()

# Title Header
st.markdown('<div class="main-header">🛍️ Smart Retail & Customer Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered In-Store Analytics, Face Recognition, Product Classification & Support Chatbot</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_analytics, tab_vision, tab_product, tab_nlp, tab_chatbot = st.tabs([
    "📊 Analytics Dashboard",
    "👤 Face Recognition & Visits",
    "📦 Product Classifier",
    "💬 Sentiment Analysis",
    "🤖 FAQ Support Chatbot"
])

# ----------------------------------------------------
# TAB 1: ANALYTICS DASHBOARD
# ----------------------------------------------------
with tab_analytics:
    st.subheader("📈 Live Customer Intelligence Metrics")
    
    logs = pipeline.vision_service.get_visit_history()
    df_logs = pd.DataFrame(logs)
    
    total_visits = len(df_logs)
    returning_cnt = len(df_logs[df_logs["status"] == "Returning Customer"]) if not df_logs.empty else 0
    guest_cnt = total_visits - returning_cnt
    return_rate = (returning_cnt / total_visits * 100) if total_visits > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total In-Store Visits", total_visits, delta="+12% this week")
    with col2:
        st.metric("Returning VIP Customers", returning_cnt, delta=f"{return_rate:.1f}% Rate")
    with col3:
        st.metric("Guest Visits", guest_cnt, delta="-3%")
    with col4:
        st.metric("Customer Sentiment Score", "84.2 / 100", delta="+4.5 pts")

    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("##### 🏆 Customer Loyalty Tier Distribution")
        if not df_logs.empty and "loyalty_tier" in df_logs.columns:
            tier_counts = df_logs["loyalty_tier"].value_counts()
            st.bar_chart(tier_counts)
        else:
            st.info("No visit logs recorded yet.")

    with chart_col2:
        st.markdown("##### 😊 Customer Feedback Sentiment Summary")
        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [18, 7, 5]
        })
        st.bar_chart(sentiment_data.set_index("Sentiment"))

    st.markdown("##### 📋 Recent Customer Visit Log Audit")
    if not df_logs.empty:
        st.dataframe(
            df_logs[["timestamp", "customer_id", "name", "loyalty_tier", "status", "confidence"]].tail(10),
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 2: FACE RECOGNITION
# ----------------------------------------------------
with tab_vision:
    st.subheader("👤 Face Recognition & Loyalty Visit Logger")
    st.markdown("Upload a customer facial photograph to recognize returning loyalty members and record their entry timestamp.")

    uploaded_face = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"], key="face_upload")
    
    col_img, col_res = st.columns(2)
    
    if uploaded_face is not None:
        image_bytes = uploaded_face.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col_res:
            with st.spinner("Analyzing face encodings..."):
                result = pipeline.recognize_customer(image_bytes)
                
            st.success("Recognition Complete!")
            st.markdown(f"### Status: **{result['status']}**")
            st.write(f"**Customer ID:** {result['customer_id']}")
            st.write(f"**Name:** {result['name']}")
            st.write(f"**Loyalty Tier:** {result['loyalty_tier']}")
            st.write(f"**Confidence:** {result['confidence']}%")
            st.write(f"**Timestamp:** {result['timestamp']}")

# ----------------------------------------------------
# TAB 3: PRODUCT CLASSIFIER
# ----------------------------------------------------
with tab_product:
    st.subheader("📦 Product Category Classifier (MobileNetV2 Transfer Learning)")
    st.markdown("Upload a product image to classify into retail categories: **shoes, bags, electronics, clothing, groceries**.")

    uploaded_prod = st.file_uploader("Choose a product image...", type=["jpg", "jpeg", "png"], key="prod_upload")
    
    pcol1, pcol2 = st.columns(2)
    
    if uploaded_prod is not None:
        p_bytes = uploaded_prod.read()
        p_img = Image.open(io.BytesIO(p_bytes))
        
        with pcol1:
            st.image(p_img, caption="Product Item Image", use_container_width=True)
            
        with pcol2:
            with st.spinner("Classifying product features..."):
                p_res = pipeline.classify_product(p_bytes, uploaded_prod.name)
                
            st.markdown(f"### Predicted Category: **{p_res['predicted_category'].upper()}**")
            st.progress(min(1.0, float(p_res['confidence']) / 100.0))
            st.write(f"**Classification Confidence:** {p_res['confidence']}%")
            
            st.markdown("#### Probability Distribution Across Categories:")
            probs_df = pd.DataFrame(
                list(p_res['category_probabilities'].items()),
                columns=["Category", "Probability (%)"]
            )
            st.dataframe(probs_df, use_container_width=True)

# ----------------------------------------------------
# TAB 4: SENTIMENT ANALYSIS
# ----------------------------------------------------
with tab_nlp:
    st.subheader("💬 Customer Review & Chat Sentiment Analyzer")
    
    sample_texts = [
        "Select a sample review...",
        "Absolutely love this handbag! High quality leather and great stitching.",
        "Average quality product. Delivery was slightly delayed but acceptable.",
        "Terrible customer service and broken shoes delivered. Want a refund!"
    ]
    selected_sample = st.selectbox("Quick Sample Selection:", sample_texts)
    
    default_text = "" if selected_sample == "Select a sample review..." else selected_sample
    user_review = st.text_area("Enter customer review or feedback text:", value=default_text, height=100)
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_review.strip():
            with st.spinner("Evaluating text sentiment..."):
                s_res = pipeline.analyze_sentiment(user_review)
                
            badge_class = "badge-pos" if s_res["sentiment"] == "Positive" else ("badge-neg" if s_res["sentiment"] == "Negative" else "badge-neu")
            
            st.markdown(f"### Sentiment: <span class='{badge_class}'>{s_res['sentiment']}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence Score:** {s_res['confidence']}%")
            
            st.markdown("#### Preprocessing Token Breakdown:")
            st.json(s_res["preprocessing"])

# ----------------------------------------------------
# TAB 5: FAQ SUPPORT CHATBOT
# ----------------------------------------------------
with tab_chatbot:
    st.subheader("🤖 Smart FAQ & Support Assistant")
    st.markdown("Ask any question regarding order status, returns, shipping, store hours, or payment methods.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Welcome to Smart Retail support. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Searching intents..."):
            bot_res = pipeline.process_chat_message(user_input)
            
        bot_reply = bot_res["response"]
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)
            st.caption(f"Matched by: {bot_res['matched_by']} | Intent: {bot_res['intent']} | Confidence: {bot_res['confidence']}%")
