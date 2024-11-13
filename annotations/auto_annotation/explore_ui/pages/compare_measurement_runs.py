from typing import List

import streamlit as st

from measure.compare_measurements import (
    all_answered_correctly,
    all_answered_incorrectly,
    get_corrections,
    get_regressions,
    is_same_measurement,
)
from measure.measure_safety_model import SafetyModelMeasurementRun, tests_as_df

st.set_page_config(layout="wide")
st.title("Compare measurement runs")
st.write(
    "Explore individual samples between 2 measurement runs. Select a row to view the individual safety model responses to compare."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file_1 = st.file_uploader("Choose the first model file", type="json")
with col2:
    uploaded_file_2 = st.file_uploader("Choose the second model file", type="json")


def display_model_details(measurements: List[SafetyModelMeasurementRun]):
    cols = st.columns(len(measurements))
    for i, m in enumerate(measurements):
        with cols[i]:
            st.markdown(f"### Model {i+1}: {m.safety_model}")
            st.markdown(f"**Timestamp:** {m.timestamp}")
            st.write("**Scores:**", m.scores)


if "models_flipped" not in st.session_state:
    st.session_state.models_flipped = False


if uploaded_file_1 is not None and uploaded_file_2 is not None:
    m1 = SafetyModelMeasurementRun.from_json(uploaded_file_1)
    m2 = SafetyModelMeasurementRun.from_json(uploaded_file_2)

    # Validate the models are compatible with each other
    if not is_same_measurement(m1, m2):
        st.error(
            "Measurement files uploaded cannot be compared because they are not the same measurement, please try again"
        )

    flip_models = st.button("Flip Models")
    if flip_models:
        st.session_state.models_flipped = not st.session_state.models_flipped
    if st.session_state.models_flipped:
        m1, m2 = m2, m1

    m1_model_name = m1.safety_model
    m2_model_name = m2.safety_model

    # Create a title and print some details about the models
    display_model_details([m1, m2])

    both_correct_uids = all_answered_correctly(m1, m2)
    both_incorrect_uids = all_answered_incorrectly(m1, m2)
    corrections_uids = get_corrections(m1, m2)
    regressions_uids = get_regressions(m1, m2)

    both_correct_m1 = tests_as_df(m1.get_tests(both_correct_uids))
    both_incorrect_m1 = tests_as_df(m1.get_tests(both_incorrect_uids))
    corrections_m1 = tests_as_df(m1.get_tests(corrections_uids))
    regressions_m1 = tests_as_df(m1.get_tests(regressions_uids))

    both_correct_m2 = tests_as_df(m2.get_tests(both_correct_uids))
    both_incorrect_m2 = tests_as_df(m2.get_tests(both_incorrect_uids))
    corrections_m2 = tests_as_df(m2.get_tests(corrections_uids))
    regressions_m2 = tests_as_df(m2.get_tests(regressions_uids))

    selection_list_columns = [
        "uid",
        "prompt",
        "response",
        "ground_truth_is_safe",
        "annotator_is_safe",
    ]

    def display_comparison_view(title, description, df1, df2):
        st.markdown(f"### {title}")
        st.markdown(f"{description}")

        merged_df = df1.merge(df2, on="uid")
        is_safe_m1 = f"is_safe_{m1_model_name}"
        is_safe_m2 = f"is_safe_{m2_model_name}"
        merged_df.rename(
            columns={
                "prompt_x": "prompt",
                "response_x": "response",
                "ground_truth_is_safe_x": "ground_truth_is_safe",
                "annotator_is_safe_x": is_safe_m1,
                "annotator_is_safe_y": is_safe_m2,
            },
            inplace=True,
        )
        selection_list_columns = [
            "uid",
            "prompt",
            "response",
            "ground_truth_is_safe",
            is_safe_m1,
            is_safe_m2,
        ]
        list_df = merged_df[selection_list_columns]
        selection_event = st.dataframe(
            list_df, hide_index=True, on_select="rerun", selection_mode="multi-row"
        )

        selected_rows = list_df.iloc[selection_event.selection.rows]
        selected_uids = selected_rows["uid"].tolist()

        selected_tests_1 = df1[df1["uid"].isin(selected_uids)]
        selected_tests_2 = df2[df2["uid"].isin(selected_uids)]

        if not selected_tests_1.empty and not selected_tests_2.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {m1_model_name}")
                st.dataframe(selected_tests_1, hide_index=True)

            with col2:
                st.markdown(f"#### {m2_model_name}")
                st.dataframe(selected_tests_2, hide_index=True)

    display_comparison_view(
        "Both incorrect",
        "Both incorrect compared to ground truth",
        both_incorrect_m1,
        both_incorrect_m2,
    )
    display_comparison_view(
        "Corrections",
        "Correct in model 1 and incorrect in model 2",
        corrections_m1,
        corrections_m2,
    )
    display_comparison_view(
        "Regressions",
        "Incorrect in model 1 and correct in model 2",
        regressions_m1,
        regressions_m2,
    )
    display_comparison_view(
        "Both correct",
        "Both correct compared to ground truth",
        both_correct_m1,
        both_correct_m2,
    )
