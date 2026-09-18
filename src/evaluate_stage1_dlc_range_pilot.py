"""Run the frozen two-model diagnostic on the multi-document Range pilot."""
import evaluate_stage1_dlc_pilot as evaluator

evaluator.DATA = evaluator.Path("artifacts/stage1-dlc-range-pilot-20260917")
evaluator.OUT = evaluator.Path("artifacts/stage1-dlc-range-pilot-evaluation-20260917")

if __name__ == "__main__":
    evaluator.main()
