from SpeeechMetric.ECAPA2_SECS import predict_SECS
from SpeeechMetric.nMOS import predict_nMOS
from SpeeechMetric.whisper_CER import predict_CER

def execute_metrics(secs=False, nmos=False, cer=False, input_file=None, ref_file=None, ground_truth=None):
    results = {}

    if secs:
        if ref_file is None:
            raise ValueError("Reference file is required for SECS metric.")
        results["SECS"] = predict_SECS(input_file, ref_file)

    if nmos:
        results["nMOS"] = predict_nMOS(input_file)

    if cer:
        if ground_truth is None:
            raise ValueError("Ground truth text is required for CER metric.")
        results["CER"] = predict_CER(input_file, ground_truth, 'vasista22/whisper-hindi-medium')

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run selected speech metrics")
    parser.add_argument("--secs", action="store_true", help="Run predict_SECS (requires reference file)")
    parser.add_argument("--nmos", action="store_true", help="Run predict_nMOS")
    parser.add_argument("--cer", action="store_true", help="Run predict_CER (requires ground truth text)")
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("--ref_file", type=str, default=None, help="Path to the reference file (only for SECS)")
    parser.add_argument("--ground_truth", type=str, default=None, help="Ground truth text for CER metric")

    args = parser.parse_args()

    try:
        result = execute_metrics(args.secs, args.nmos, args.cer, args.input_file, args.ref_file, args.ground_truth)
        print(result)
    except ValueError as e:
        print(f"Error: {e}")
