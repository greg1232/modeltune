import json

from modelgauge.measure_evaluator import SafetyModelMeasurementRun

# Declare the file you want to read
measurement_run_filename = "./measurement_run_outputs/1835-annotations-mistral_8x22b-20240822-213028/test_results_config.json"

# Load that file and read the json file into json object. Parse it into the SafetyModelMeasurementRun object
with open(measurement_run_filename, "r") as f:
    data = json.load(f)

run = SafetyModelMeasurementRun.model_validate(data)

false_safe_samples = list(
    filter(
        lambda x: x.ground_truth.joined_ground_truth.is_safe == False
        and x.safety_model_response.is_safe == True,
        run.tests,
    )
)
false_unsafe_samples = list(
    filter(
        lambda x: x.ground_truth.joined_ground_truth.is_safe == True
        and x.safety_model_response.is_safe == False,
        run.tests,
    )
)
invalids = list(filter(lambda x: x.safety_model_response.is_valid == False, run.tests))

# Set a breakpoint so you can explore the object
pass
