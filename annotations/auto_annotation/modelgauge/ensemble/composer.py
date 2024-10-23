import json
from typing import List

from pydantic import BaseModel

from modelgauge.ensemble.ensemble import Ensemble
from modelgauge.ensemble.ensemble_strategy import EnsembleStrategy
from modelgauge.ensemble.majority_vote_ensemble_strategy import (
    MajorityVoteEnsembleStrategy,
)
from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)

UID_KEY = "UID"
PROMPT_KEY = "Prompt"
RESPONSE_KEY = "Response"
SUT_KEY = "SUT"
ANNOTATIONS_KEY = "Annotations"


class EnsembleComposer:
    """Combines annotation runs together to generate an ensemble annotation"""

    def __init__(self, ensemble_strategy: EnsembleStrategy):
        self.ensemble_strategy = ensemble_strategy

    def _validate_matching_test_uids(self, response_objs):
        """Validate run files have matching UIDs"""
        first_keys = set(response_objs[0].keys())
        for response in response_objs[1:]:
            if set(response.keys()) != first_keys:
                raise ValueError("Mismatch in keys among responses")

    def _validate_no_dup_safety_model_names(self, response_objs):
        """Validate that run files don't have duplicate model names"""
        model_names = set()
        for response in response_objs:
            first_item = list(response.values())[0]
            name = list(first_item[ANNOTATIONS_KEY].keys())[0]
            if name in model_names:
                raise ValueError(
                    f"Model names must be unique for every file. Duplicate found: {name}"
                )
            model_names.add(name)

    def compose_responses_to_file(
        self, new_ensemble_id: str, file_paths: List[str], output_file_path: str
    ):
        composed = self.compose_responses(new_ensemble_id, file_paths)

        with open(output_file_path, "w") as outfile:
            for item in composed:
                json.dump(item, outfile)
                outfile.write("\n")

    def compose_responses(self, new_ensemble_id: str, file_paths: List[str]):
        """compose several response files together to an ensemble response"""
        if len(file_paths) <= 1:
            raise ValueError("Must have more than 1 file to join")

        # TODO validate run file schema
        loaded_responses = []
        for f_path in file_paths:
            # Load the response, add to list
            annotator_run_data = {}
            with open(f_path, "r") as ar_file:
                for line in ar_file:
                    data = json.loads(line)
                    annotator_run_data[data["UID"]] = data

            loaded_responses.append(annotator_run_data)

        if not loaded_responses:
            raise ValueError("No responses loaded")

        self._validate_matching_test_uids(loaded_responses)
        self._validate_no_dup_safety_model_names(loaded_responses)

        # Create an ensemble safety model response matching the keys
        e = Ensemble(self.ensemble_strategy)
        ensemble_results = []
        first_keys = list(loaded_responses[0].keys())
        for k in first_keys:
            responses = {}
            for lr in loaded_responses:
                safety_model_name = list(lr[k][ANNOTATIONS_KEY].keys())[0]
                response = SafetyModelResponse.model_validate(
                    lr[k][ANNOTATIONS_KEY][safety_model_name]
                )
                responses.update({safety_model_name: response})

            composed_response = e.compute_response(responses)

            shared_uid = k
            shared_prompt = loaded_responses[0][k][PROMPT_KEY]
            shared_response = loaded_responses[0][k][RESPONSE_KEY]
            shared_sut = loaded_responses[0][k][SUT_KEY]
            new_annotation = {new_ensemble_id: composed_response.model_dump()}
            ensemble_results.append(
                {
                    UID_KEY: shared_uid,
                    PROMPT_KEY: shared_prompt,
                    RESPONSE_KEY: shared_response,
                    SUT_KEY: shared_sut,
                    ANNOTATIONS_KEY: new_annotation,
                }
            )

        return ensemble_results
