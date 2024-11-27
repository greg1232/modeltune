import argparse

from modelgauge.ensemble.composer import EnsembleComposer
from modelgauge.ensemble.ensemble_strategy import EnsembleStrategies
from modelgauge.ensemble.majority_vote_ensemble_strategy import (
    MajorityVoteSafeTiesEnsembleStrategy,
    MajorityVoteUnsafeTiesEnsembleStrategy,
)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool that composes multiple annotator run files into an ensemble annotation file"
    )
    parser.add_argument(
        "annotation_files",
        metavar="annotation_files",
        type=str,
        nargs="+",
        help="a list of annotator run files",
    )
    parser.add_argument(
        "-s", "--ensemble_strategy", type=str, help="the ensemble strategy to use"
    )
    parser.add_argument("-i", "--new_id", type=str, help="the new ensemble ID")
    parser.add_argument(
        "-o", "--output_filename", type=str, help="the JSONL output filename"
    )

    args = parser.parse_args()

    print("Mixing the following files:")
    for file in args.annotation_files:
        print(file)
    print("\n")

    # Instantiate the appropriate ensemble strategy
    if args.ensemble_strategy == EnsembleStrategies.MAJORITY_TIE_UNSAFE:
        print("Using majority vote unsafe tiebreaker strategy to mix results")
        strategy = MajorityVoteUnsafeTiesEnsembleStrategy()
    elif args.ensemble_strategy == EnsembleStrategies.MAJORITY_TIE_SAFE:
        print("Using majority vote safe tiebreaker strategy to mix results")
        strategy = MajorityVoteSafeTiesEnsembleStrategy()
    else:
        raise ValueError(f"Unknown ensemble strategy: {args.ensemble_strategy}")

    # Create the EnsembleComposer with the chosen strategy
    composer = EnsembleComposer(ensemble_strategy=strategy)

    # Compose the responses to the output file
    print("Running ensemble composer...")
    composer.compose_responses_to_file(
        new_ensemble_id=args.new_id,
        file_paths=args.annotation_files,
        output_file_path=args.output_filename,
    )
    print("Done!")
    print(f"New output file available: {args.output_filename}")


if __name__ == "__main__":
    main()
