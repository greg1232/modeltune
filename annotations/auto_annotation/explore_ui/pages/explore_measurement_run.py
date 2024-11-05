import pandas as pd
import streamlit as st

from measure.measure_safety_model import SafetyModelMeasurementRun, tests_as_df

st.title("Single Measurement Run Explorer")

uploaded_file = st.file_uploader("Choose a measurement run file", type="json")
if uploaded_file is not None:
    m = SafetyModelMeasurementRun.from_json(uploaded_file)

    st.markdown(f"## Measurement Run: {m.run_id}")

    st.markdown(f"Safety model: {m.safety_model}")

    st.markdown(f"### Scores")
    st.write(m.scores)

    st.markdown(f"### False safes")
    st.dataframe(tests_as_df(m.get_false_safes()))

    st.markdown("### False unsafes")
    st.dataframe(tests_as_df(m.get_false_unsafes()))

    st.markdown("### Invalids")
    st.dataframe(tests_as_df(m.get_invalids()))
