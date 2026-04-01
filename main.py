import tyro

from pipeline import Pipeline, PipelineConfig
from evaluate import evaluate_all_results
from utils import print_header, print_section, print_info, print_success


def main():
    config = tyro.cli(PipelineConfig)

    print_header("NVDA 9:30-10:00 RV Prediction Pipeline")
    print_info(f"Config: {config}")

    pipeline = Pipeline(config)

    print_section("Running Pipeline")
    full_results, selected_results, selected_features = pipeline.run()

    print_section("Evaluating Results")
    full_metrics, selected_metrics = evaluate_all_results(
        full_results,
        selected_results,
        pipeline.y_test.values,
        selected_features,
        pipeline.feature_names,
    )

    print_success("Pipeline complete")


if __name__ == "__main__":
    main()
