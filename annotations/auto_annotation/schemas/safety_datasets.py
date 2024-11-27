import uuid
import warnings
from abc import ABC
from collections import Counter
from collections.abc import Iterator
from typing import Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel

from modelgauge.ensemble.ensemble import Ensemble
from modelgauge.ensemble.majority_vote_ensemble_strategy import (
    MajorityVoteUnsafeTiesEnsembleStrategy,
)
from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)
from modelgauge.single_turn_prompt_response import Annotation


def required_fields(model: type[BaseModel], recursive: bool = False) -> Iterator[str]:
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        t = field.annotation
        if recursive and isinstance(t, type) and issubclass(t, BaseModel):
            yield from required_fields(t, recursive=True)
        else:
            yield name


class BaseSafetyDataset(ABC):
    """Base safety dataset class with loader and exporter methods"""

    # Ideally this is a required abstract property
    schema: Type[BaseModel]

    def __init__(self, df: pd.DataFrame):
        required_columns = required_fields(self.schema)
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
        self.df = df

    def __len__(self):
        return len(self.df)

    @classmethod
    def from_csv(cls, filepath: str):
        """Load safety model dataset from csv"""
        df = pd.read_csv(filepath)

        return cls(df=df)

    def to_csv(self, output_path: str):
        """Write to csv at output_path"""
        self.df.to_csv(output_path, index=False)


class AnnotationInputDatasetSchema(BaseModel):
    """Dataset schema for input to modelgauge annotation"""

    uid: str = "UID"
    sut: str = "SUT"
    prompt: str = "Prompt"
    response: str = "Response"


class AnnotationInputDataset(BaseSafetyDataset):
    """Dataset for input to modelgauge annotation"""

    schema: Type[BaseModel] = AnnotationInputDatasetSchema

    def ref_schema(self):
        return AnnotationInputDatasetSchema()

    def uids(self):
        return self.df[self.ref_schema().uid]

    def get_subset(self, uids: list[str]) -> "AnnotationInputDataset":
        """Return a subset annotation input dataset"""
        missing_uids = [
            uid for uid in uids if uid not in self.df[self.ref_schema().uid].values
        ]
        if missing_uids:
            raise ValueError(
                f"The following UIDs are not present in the dataset: {missing_uids}"
            )

        subset_df = self.df[self.df[self.ref_schema().uid].isin(uids)]
        return AnnotationInputDataset(df=subset_df)


class PromptDatasetSchema(BaseModel):
    """Dataset schema tracking prompts"""

    prompt_uid: str = "prompt_uid"
    hazard: Optional[str] = "hazard"
    hazard_subtype: Optional[str] = "hazard_subtype"
    prompt_text: str = "prompt_text"


class PromptDataset(BaseSafetyDataset):
    """Dataset tracking prompts"""

    schema: Type[BaseModel] = PromptDatasetSchema


class ResponseDatasetSchema(PromptDatasetSchema):
    """Dataset schema tracking prompts and responses"""

    sut_uid: str = "sut_uid"
    pair_uid: str = "pair_uid"
    response_text: str = "response_text"


class ResponseDataset(BaseSafetyDataset):
    schema: Type[BaseModel] = ResponseDatasetSchema

    def to_annotation_dataset(self) -> AnnotationInputDataset:
        response_dataset_schema = ResponseDatasetSchema()
        annotator_input_schema = AnnotationInputDatasetSchema()

        mapper = {
            response_dataset_schema.pair_uid: annotator_input_schema.uid,
            response_dataset_schema.prompt_text: annotator_input_schema.prompt,
            response_dataset_schema.response_text: annotator_input_schema.response,
            response_dataset_schema.sut_uid: annotator_input_schema.sut,
        }

        out_df = self.df.rename(columns=mapper)
        out_df = out_df[
            [
                annotator_input_schema.uid,
                annotator_input_schema.prompt,
                annotator_input_schema.response,
                annotator_input_schema.sut,
            ]
        ]

        annotation_input = AnnotationInputDataset(df=out_df)
        return annotation_input


class SafetyModelAnnotationDatasetSchema(BaseModel):
    pair_uid: str = "pair_uid"
    prompt_text: str = "prompt_text"
    response_text: str = "response_text"
    sut: str = "sut"
    annotation_uid: str = "annotation_uid"
    annotator: str = "annotator"
    is_safe: str = "is_safe"
    safety_categories: str = "safety_categories"
    reasoning: str = "reasoning"
    is_valid: str = "is_valid"
    raw_response: str = "raw_response"
    logprobs: str = "logprobs"
    total_tokens_used: str = "total_tokens_used"


class SafetyModelAnnotationDataset(BaseSafetyDataset):
    schema: Type[BaseModel] = SafetyModelAnnotationDatasetSchema

    @classmethod
    def concat(
        cls, datasets: List["SafetyModelAnnotationDataset"]
    ) -> "SafetyModelAnnotationDataset":
        dataset_dfs = list(map(lambda x: x.df, datasets))
        df = pd.concat(dataset_dfs, ignore_index=True)
        return cls(df)

    def get_unsafes(self):
        schema = SafetyModelAnnotationDatasetSchema()
        return self.df[self.df[schema.is_safe] == False]

    def get_safes(self):
        schema = SafetyModelAnnotationDatasetSchema()
        return self.df[self.df[schema.is_safe]]

    def _row_to_safety_model_response(self, row):
        annotation_dataset_schema = SafetyModelAnnotationDatasetSchema()
        return SafetyModelResponse(
            is_safe=row[annotation_dataset_schema.is_safe],
            safety_categories=row[annotation_dataset_schema.safety_categories],
            reasoning=row[annotation_dataset_schema.reasoning],
            raw_response=row[annotation_dataset_schema.raw_response],
            is_valid=row[annotation_dataset_schema.is_valid],
            logprobs=row[annotation_dataset_schema.logprobs],
            total_tokens_used=row[annotation_dataset_schema.total_tokens_used],
        )

    def compute_pseudo_ground_truths(self):
        """Generate a column of ground truths by computing across the annotator UIDs present"""

        new_obj_col = "safety_model_response"

        responses_obj_df = self.df.copy()
        responses_obj_df[new_obj_col] = responses_obj_df.apply(
            self._row_to_safety_model_response, axis=1
        )

        annotation_dataset_schema = SafetyModelAnnotationDatasetSchema()

        # Check for duplicates in pair_uid and drop them
        if responses_obj_df.duplicated(
            subset=[
                annotation_dataset_schema.pair_uid,
                annotation_dataset_schema.annotator,
            ]
        ).any():
            warnings.warn(
                "Duplicates found in the combination of pair_uid and annotator columns. Dropping duplicates."
            )
            responses_obj_df = responses_obj_df.drop_duplicates(
                subset=[
                    annotation_dataset_schema.pair_uid,
                    annotation_dataset_schema.annotator,
                ]
            )

        pivoted = responses_obj_df.pivot(
            index=annotation_dataset_schema.pair_uid,
            columns=annotation_dataset_schema.annotator,
            values=new_obj_col,
        )

        # TODO validate for every pair_uid, there exists an annotation from every annotator

        # compute pseudo ground truth column with output
        e = Ensemble(strategy=MajorityVoteUnsafeTiesEnsembleStrategy())

        def get_combined_response(row) -> EnsembleSafetyModelResponse:
            responses = {}
            for annotator in row.index:
                responses[annotator] = row[annotator]
            response = e.compute_response(responses)
            return response

        def get_agreement_strength(joint_annotation: EnsembleSafetyModelResponse):
            unanimous = "unanimous"
            majority = "majority"
            evensplit = "evensplit"

            is_safes = [x.is_safe for x in joint_annotation.joined_responses.values()]
            counts = Counter(is_safes)

            if counts[True] == counts[False]:
                return evensplit
            elif counts[True] == len(is_safes) or counts[False] == len(is_safes):
                return unanimous
            else:
                return majority

        p_gt_schema = PseudoGroundTruthDatasetSchema()
        combined_obj_col = "combined_response_obj"
        pivoted[combined_obj_col] = pivoted.apply(get_combined_response, axis=1)
        pivoted[p_gt_schema.pseudo_gt_is_safe] = pivoted[combined_obj_col].apply(
            lambda x: x.is_safe
        )
        pivoted[p_gt_schema.pseudo_gt_hazard_categories] = pivoted[
            combined_obj_col
        ].apply(lambda x: x.safety_categories)
        pivoted[p_gt_schema.pseudo_gt_agreement_strength] = pivoted[
            combined_obj_col
        ].apply(get_agreement_strength)

        # output the new data structure (might need to create it)
        pseudo_gt_df = pivoted.loc[
            :,
            [
                p_gt_schema.pseudo_gt_is_safe,
                p_gt_schema.pseudo_gt_agreement_strength,
                p_gt_schema.pseudo_gt_hazard_categories,
            ],
        ].reset_index()

        # TODO something weird happening with multilevel index and presence of "annotator"
        pseudo_gt = PseudoGroundTruthDataset(pseudo_gt_df)

        return pseudo_gt


class PseudoGroundTruthDatasetSchema(BaseModel):
    pair_uid: str = "pair_uid"
    pseudo_gt_is_safe: str = "pseudo_gt_is_safe"
    pseudo_gt_agreement_strength: str = "pseudo_gt_agreement_strength"
    pseudo_gt_hazard_categories: str = "pseudo_gt_hazard_categories"


class PseudoGroundTruthDataset(BaseSafetyDataset):
    schema: Type[BaseModel] = PseudoGroundTruthDatasetSchema


class AnnotationRunSchema(BaseModel):
    uid: str = "UID"
    prompt: str = "Prompt"
    response: str = "Response"
    sut: str = "SUT"
    annotations: str = "Annotations"


class AnnotationRun:
    """Class representing annotation run data"""

    def __init__(self, df: pd.DataFrame):
        # Validate df before assigning it
        self._validate_df(df)
        self.df: pd.DataFrame = df
        self.schema_ref = AnnotationRunSchema()

    def to_jsonl(self, filepath: str):
        """Write to jsonl file"""
        self.df.to_json(filepath, orient="records", lines=True)
        pass

    @classmethod
    def from_jsonl(cls, filepath: str):
        df = pd.read_json(filepath, lines=True)
        return cls(df)

    def __len__(self):
        return len(self.df)

    def uids(self):
        return self.df[self.schema_ref.uid]

    def get_safes(self):
        sdf = self.to_safety_model_annotation()
        return sdf.get_safes()

    def get_unsafes(self):
        sdf = self.to_safety_model_annotation()
        return sdf.get_unsafes()

    def _validate_df(self, df: pd.DataFrame):
        # TODO not implemented
        return True

    def to_safety_model_annotation(self) -> SafetyModelAnnotationDataset:
        """Output"""
        # TODO just read the annotation into a json column
        src_schema = AnnotationRunSchema()
        target_schema = SafetyModelAnnotationDatasetSchema()

        column_mapper = {
            src_schema.uid: target_schema.pair_uid,
            src_schema.prompt: target_schema.prompt_text,
            src_schema.response: target_schema.response_text,
            src_schema.sut: target_schema.sut,
        }

        out_df = self.df.rename(columns=column_mapper)

        # Break down the annotation object into columns then drop it
        annotations = out_df[src_schema.annotations]

        out_df[target_schema.annotation_uid] = [
            str(uuid.uuid4()) for _ in range(len(out_df))
        ]
        out_df[target_schema.annotator] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[0]
        )
        out_df[target_schema.is_safe] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].is_safe
        )
        out_df[target_schema.safety_categories] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].safety_categories
        )
        out_df[target_schema.reasoning] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].reasoning
        )
        out_df[target_schema.raw_response] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].raw_response
        )
        out_df[target_schema.is_valid] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].is_valid
        )
        out_df[target_schema.logprobs] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].logprobs
        )
        out_df[target_schema.total_tokens_used] = annotations.map(
            lambda x: self._parse_annotation_obj(x)[1].total_tokens_used
        )

        out_df.drop(columns=[src_schema.annotations], inplace=True)

        result = SafetyModelAnnotationDataset(out_df)

        return result

    def _parse_annotation_obj(
        self, annotations: Dict[str, object]
    ) -> tuple[str, SafetyModelResponse]:
        keys = list(annotations.keys())
        if len(keys) != 1:
            raise ValueError(
                f"Number of annotations must be 1. Any other number is unsupported. Number of annotations found: {len(keys)}"
            )

        annotator_id = keys[0]
        response = SafetyModelResponse.model_validate(annotations[annotator_id])
        return (annotator_id, response)

    def matches_annotation_input_dataset(self, input_dataset: AnnotationInputDataset):
        input_dataset_uids = input_dataset.uids().sort_values().reset_index(drop=True)
        annotation_uids = (
            self.df[self.schema_ref.uid].sort_values().reset_index(drop=True)
        )

        if not input_dataset_uids.equals(annotation_uids):
            return False
        else:
            return True

    def merge(self, other_run: "AnnotationRun"):
        """Merge with another AnnotationRun"""
        # Concatenate the dataframes
        merged_df = pd.concat([self.df, other_run.df], ignore_index=True)

        # Return a new AnnotationRun instance with the concatenated dataframe
        return AnnotationRun(df=merged_df)


class MeasurementGroundTruthDatasetSchema(BaseModel):
    pair_uid: str = "pair_uid"
    prompt_text: str = "prompt_text"
    response_text: str = "response_text"
    sut_uid: str = "sut_uid"
    hazard: str = "hazard"
    human_annotation_1: str = "human_annotation_1"
    human_annotation_2: str = "human_annotation_2"
    human_annotation_3: str = "human_annotation_3"
    human_unanimous_agreement: str = "human_unanimous_agreement"
    ground_truth_label: str = "ground_truth_label"


class MeasurementGroundTruthDataset(BaseSafetyDataset):
    schema = MeasurementGroundTruthDatasetSchema

    def to_responses_dataset(self):
        return ResponseDataset(self.df)

    def write_annotation_input_csv(self, output_path: str):
        src_schema = MeasurementGroundTruthDatasetSchema()
        annotator_input_schema = AnnotationInputDatasetSchema()

        mapper = {
            src_schema.pair_uid: annotator_input_schema.uid,
            src_schema.prompt_text: annotator_input_schema.prompt,
            src_schema.response_text: annotator_input_schema.response,
            src_schema.sut_uid: annotator_input_schema.sut,
        }

        out_df = self.df.rename(columns=mapper)
        out_df = out_df[
            [
                annotator_input_schema.uid,
                annotator_input_schema.prompt,
                annotator_input_schema.response,
                annotator_input_schema.sut,
            ]
        ]

        out_df.to_csv(output_path, index=False)

    def is_compatible_with_run(self, run: AnnotationRun):
        self_schema = MeasurementGroundTruthDatasetSchema()
        run_uids = run.uids().sort_values().reset_index(drop=True)
        gt_uids = self.df[self_schema.pair_uid].sort_values().reset_index(drop=True)

        if not run_uids.equals(gt_uids):
            return False
        else:
            return True


class MeasurementResultsDatasetSchema(MeasurementGroundTruthDatasetSchema):
    """Measurement results dataset schema. Annotator can mean an individual annotator or an ensemble"""

    annotator: str = "annotator"
    annotator_is_safe: str = "annotator_is_safe"
    annotator_safety_categories: str = "annotator_safety_categories"
    annotator_reasoning: str = "annotator_reasoning"
    annotator_is_valid: str = "annotator_is_valid"
    annotator_raw_response: str = "annotator_raw_response"


class MeasurementResultsDataset(BaseSafetyDataset):
    schema: Type[BaseModel] = MeasurementResultsDatasetSchema
