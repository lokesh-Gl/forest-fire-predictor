import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# === Streamlit Page Config ===
st.set_page_config(page_title="Wild Flame Watcher")
st.markdown(
    "<h1 style='text-align: center; color:#cbcbcb; font-size: 43px; font-family: Geneva;'>Forest Fire Risk Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color:#e2e5de; font-size: 22px;'>Detect Risk. Prevent Fires. Save Forests.</p>",
    unsafe_allow_html=True
)

# === Load Model & Features ===
try:
    model = joblib.load("random_forest_fire_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f"Error loading model or feature columns: {e}")
    st.stop()

# === Session State Initialization ===
if "user_input" not in st.session_state:
    st.session_state.user_input = {feature: "" for feature in feature_cols}
if "predicted_location" not in st.session_state:
    st.session_state.predicted_location = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "location_name" not in st.session_state:
    st.session_state.location_name = ""
if "fire_intensity_level" not in st.session_state:
    st.session_state.fire_intensity_level = ""


# === Collect User Input ===
st.markdown(
    "<h4 style='text-align: left; color:#e2e5de; font-weight: 500; margin-top: 20px;'>Fill in Current Conditions</h4>",
    unsafe_allow_html=True
)
input_valid = True

for feature in feature_cols:
    default_value = str(st.session_state.user_input.get(feature, ""))

    if feature.lower() in ["date", "day", "month", "year"]:
        user_val = st.text_input(f"{feature}", value=default_value)
        if user_val and not user_val.isdigit():
            st.warning(f"⚠️ Please enter only digits for {feature}")
            input_valid = False
        else:
            st.session_state.user_input[feature] = user_val
    else:
        user_val = st.text_input(f"{feature}", value=default_value)
        try:
            float(user_val)
            st.session_state.user_input[feature] = user_val
        except ValueError:
            if user_val != "":
                st.warning(f"⚠️ Please enter a valid number for {feature}")
                input_valid = False

# === Predict Button ===
if st.button("Predict Fire Risk"):
    user_input_filled = all(st.session_state.user_input[feat] != "" for feat in feature_cols)

    if not user_input_filled:
        st.warning("⚠️ Please fill in all values before predicting.")
    elif not input_valid:
        st.warning("⚠️ Invalid input detected. Please correct the values.")
    else:
        input_data = {
            key: int(val) if key.lower() in ["date", "day", "month", "year"]
            else float(val)
            for key, val in st.session_state.user_input.items()
        }

        input_df = pd.DataFrame([input_data])
        prediction = str(model.predict(input_df)[0])
        st.session_state.prediction_result = prediction

        # === Fire Intensity Logic (Only if fire is predicted) ===
        if prediction.lower() == "fire":
            ffmc = input_data.get("FFMC", 0)
            dmc = input_data.get("DMC", 0)
            dc = input_data.get("DC", 0)
            isi = input_data.get("ISI", 0)
            temp = input_data.get("temp", 0)
            rh = input_data.get("RH", 100)
            wind = input_data.get("wind", 0)

            # Simple logic: tweak thresholds as needed
            score = (
                0.3 * ffmc +
                0.2 * dmc +
                0.15 * dc +
                0.1 * isi +
                0.1 * temp +
                0.1 * wind -
                0.05 * rh
            )

            if score < 100:
                intensity = "Low"
            elif 100 <= score < 200:
                intensity = "Medium"
            else:
                intensity = "High"

            st.session_state.fire_intensity_level = intensity

        # === Get Location from Lat/Lon ===
        lat = input_data.get("latitude", 20.5937)
        lon = input_data.get("longitude", 78.9629)
        st.session_state.predicted_location = [lat, lon]

        try:
            geolocator = Nominatim(user_agent="fire_predictor")
            location = geolocator.reverse(f"{lat}, {lon}", language="en")
            st.session_state.location_name = location.address if location else "Unknown location"
        except:
            st.session_state.location_name = "Unknown location"

        # === Show Prediction Result ===
        if prediction.lower() == "fire":
            st.error(" High Fire Risk Detected!")
            st.info(f" **Fire Intensity Level:** {st.session_state.fire_intensity_level}")
        else:
            st.success(" No Fire Expected.")

# === Display Map After Prediction ===
if st.session_state.predicted_location:
    st.subheader(" Predicted Fire Location Map")
    location = st.session_state.predicted_location
    m = folium.Map(location=location, zoom_start=7)
    folium.Marker(location, popup=" Predicted Fire Spot").add_to(m)
    st_folium(m, width=700)

    st.markdown(f"** Location:** {st.session_state.location_name}")
    st.markdown(f"** Prediction Result:** {st.session_state.prediction_result}")
    if st.session_state.fire_intensity_level:
        st.markdown(f"**⚡ Fire Intensity:** {st.session_state.fire_intensity_level}")

# === Reset Inputs ===
if st.button("Reset Inputs"):
    st.session_state.user_input = {feature: "" for feature in feature_cols}
    st.session_state.predicted_location = None
    st.session_state.prediction_result = None
    st.session_state.location_name = ""
    st.session_state.fire_intensity_level = ""
    st.rerun()

# === Feature Importance Plot ===
if st.checkbox("Show Feature Importance"):
    st.subheader("Feature Importance (Random Forest)")
    importances = model.feature_importances_
    features = feature_cols
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)

# === Historical Trends ===
if st.checkbox("Show Past Fire Trends"):
    st.subheader("Past Fire Records Over Years")
    try:
        df = pd.read_csv("forest_fire_updated.csv")
        fire_trends = df.groupby("year")["Classes"].value_counts().unstack().fillna(0)
        fig = px.bar(
            fire_trends,
            title="Fire vs No Fire per Year",
            barmode="group",
            labels={"value": "Number of Incidents", "year": "Year"},
            text_auto=True,
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# --> Note Of Details <--

if st.checkbox("Details Of Values"):
    st.markdown("""
    1. FFMC – Fine Fuel Moisture Code
    - **Purpose:** Indicates the moisture content of surface litter and fine fuels (leaves, small twigs).
    - **Range:** 0 to 101+ (higher = drier fuels)
    - **Significance:**  
        - A high FFMC means fine fuels are dry and can ignite easily.
        - Sensitive to wind, temperature, humidity, and precipitation.

    2. DMC – Duff Moisture Code
    - **Purpose:** Represents the moisture content in loosely compacted organic layers (duff) below the surface.
    - **Range:** 0 to 150+ (higher = drier conditions)
    - **Significance:**  
        - Affects intermediate fuel layers.  
        - Important for sustained fire behavior and fuel consumption.

    3. ISI – Initial Spread Index
    - **Purpose:** Combines FFMC and wind speed to predict the rate of fire spread.
    - **Range:** No strict upper limit (usually 0 to 50+)
    - **Significance:**  
        - High ISI = fast-moving fires.  
        - Used for short-term firefighting decisions.

    4. FWI – Fire Weather Index
    - **Purpose:** Overall measure of fire danger, calculated from ISI and BUI.
    - **Range:** 0 to 100+ (or more depending on conditions)
    - **Significance:**  
        - A higher FWI means a more intense and dangerous fire.
    """)