import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Safe XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

st.set_page_config(page_title="Fatigue AI Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1517649763962-0c623066013b");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

st.title("🏃 Fatigue Detection Dashboard")

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("multimodal_sports_injury_dataset.csv")

df = load_data().copy()

# --------------------------
# CLEAN DATA
# --------------------------
for col in df.select_dtypes(include=np.number):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode()[0], inplace=True)

# --------------------------
# TARGET
# --------------------------
threshold = df['fatigue_index'].quantile(0.7)
df['fatigue_label'] = (df['fatigue_index'] > threshold).astype(int)

# --------------------------
# FEATURES
# --------------------------
features = [
    'heart_rate','body_temperature','hydration_level','sleep_quality',
    'recovery_score','stress_level','training_load','training_duration',
    'muscle_activity','step_count','ground_reaction_force'
]

X = df[features]
y = df['fatigue_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# TRAIN MODELS (FIXED RF)
# --------------------------
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42
        )
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss')

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })

    return models, pd.DataFrame(results)

models, results_df = train_models()
results_df = results_df.sort_values(by="F1 Score", ascending=False)

# 🔥 FORCE BEST MODEL (IMPORTANT)
best_model = models["Random Forest"]

# --------------------------
# INPUT
# --------------------------
st.sidebar.header("Athlete Input")

input_data = {
    "heart_rate": st.sidebar.slider("Heart Rate", 50, 200, 80),
    "body_temperature": st.sidebar.slider("Body Temperature", 35.0, 42.0, 37.0),
    "hydration_level": st.sidebar.slider("Hydration Level", 0, 100, 70),
    "sleep_quality": st.sidebar.slider("Sleep Quality", 0, 100, 80),
    "recovery_score": st.sidebar.slider("Recovery Score", 0, 100, 60),
    "stress_level": st.sidebar.slider("Stress Level", 0, 100, 50),
    "training_load": st.sidebar.slider("Training Load", 0, 500, 200),
    "training_duration": st.sidebar.slider("Training Duration", 0, 180, 60),
    "muscle_activity": st.sidebar.slider("Muscle Activity", 0, 100, 50),
    "step_count": st.sidebar.slider("Step Count", 0, 20000, 5000),
    "ground_reaction_force": st.sidebar.slider("Ground Force", 0, 5000, 1000)
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df.astype(float))

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "ROC Curve"])

# --------------------------
# TAB 1: PREDICTION (FIXED)
# --------------------------
with tab1:
    st.subheader("Live Prediction")

    if st.button("Predict Fatigue"):
        prob = best_model.predict_proba(input_scaled)[0][1]

        st.write(f"Fatigue Probability: {prob:.4f}")

        if prob > 0.2:   # 🔥 Lower threshold
            st.error("⚠️ High Fatigue")
        else:
            st.success("✅ Low Fatigue")

    # 🔥 FIXED RADAR CHART (REAL-TIME)
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(input_data.values()),
        theta=list(input_data.keys()),
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# TAB 2
# --------------------------
with tab2:
    st.subheader("Model Comparison")

    st.dataframe(results_df)

    st.plotly_chart(
        px.bar(results_df, x="Model", y=["Accuracy","Precision","Recall","F1 Score"], barmode="group"),
        use_container_width=True
    )

# --------------------------
# TAB 3
# --------------------------
with tab3:
    st.subheader("ROC Curve")

    fig = go.Figure()

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{name} (AUC={roc_auc:.2f})"
            ))

    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        line=dict(dash='dash'),
        name="Random"
    ))

    st.plotly_chart(fig, use_container_width=True)