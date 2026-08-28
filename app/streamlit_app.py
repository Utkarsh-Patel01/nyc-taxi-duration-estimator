import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import InvalidTripInputError, load_model, predict

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_final.joblib"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


st.set_page_config(
    page_title="NYC Taxi ETA Estimator", page_icon="🚕", layout="centered"
)
st.title("🚕 NYC Taxi Trip Duration Estimator")
st.caption(
    "Estimates trip duration from pickup/dropoff location and time, "
    "trained on 1.4M+ real NYC taxi trips (tuned LightGBM)."
)

model = get_model()

with st.form("trip_form"):
    st.subheader("Pickup")
    col1, col2 = st.columns(2)
    with col1:
        pickup_lat = st.number_input("Pickup latitude", value=40.7580, format="%.6f")
    with col2:
        pickup_lon = st.number_input("Pickup longitude", value=-73.9855, format="%.6f")

    st.subheader("Dropoff")
    col3, col4 = st.columns(2)
    with col3:
        dropoff_lat = st.number_input("Dropoff latitude", value=40.7484, format="%.6f")
    with col4:
        dropoff_lon = st.number_input("Dropoff longitude", value=-73.9857, format="%.6f")

    st.subheader("Trip details")
    col5, col6 = st.columns(2)
    with col5:
        pickup_date = st.date_input("Pickup date", value=date(2016, 3, 14))
    with col6:
        pickup_time = st.time_input("Pickup time", value=dtime(18, 30))

    col7, col8 = st.columns(2)
    with col7:
        passenger_count = st.slider("Passenger count", min_value=1, max_value=6, value=1)
    with col8:
        vendor_id = st.selectbox("Vendor", options=[1, 2], index=0)

    submitted = st.form_submit_button("Estimate Trip Duration", type="primary")

if submitted:
    pickup_datetime = datetime.combine(pickup_date, pickup_time)
    try:
        result = predict(
            model,
            pickup_latitude=pickup_lat,
            pickup_longitude=pickup_lon,
            dropoff_latitude=dropoff_lat,
            dropoff_longitude=dropoff_lon,
            pickup_datetime=pickup_datetime,
            passenger_count=passenger_count,
            vendor_id=vendor_id,
        )
        st.success(result["duration_formatted"])
        st.caption(f"Raw prediction: {result['duration_seconds']:.0f} seconds")
    except InvalidTripInputError as error:
        st.error(f"Invalid trip input: {error}")
