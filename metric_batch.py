from SpeeechMetric.ECAPA2_SECS import predict_SECS_batch
from SpeeechMetric.nMOS import predict_nMOS_batch
from SpeeechMetric.whisper_CER import predict_CER
import pandas as pd


def extract_file_paths(csv_path):
    df = pd.read_csv(csv_path)

    file_names = df["file_name"].tolist()
    ref_files = df["ref_file"].tolist()
    ground_truths = df["ground_truth"].tolist()
    return file_names, ground_truths, ref_files


def write_results_to_csv(results, output_csv):
    # Create a DataFrame from the results
    data = {
        "file_name": results["file_names"],
        "SECS": results["SECS"],
        "nMOS": results["nMOS"],
        "CER": results["CER"],
    }

    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

def execute_metrics(secs=False, nmos=False, cer=False, input_csv=None, output_csv=None):

    file_names, ground_truths, ref_files = extract_file_paths(input_csv)

    results = {
        "file_names":file_names,
        "SECS":[],
        "nMOS": [],
        "CER": []
    }

    if secs:
        assert len(file_names)==len(ref_files), "Number of reference files should be same as number of files"
        results["SECS"] = predict_SECS_batch(file_names, ref_files)
    else:
        results["SECS"] = [None]*len(file_names)

    if nmos:
        results["nMOS"] = predict_nMOS_batch(file_names)
    else:
        results["nMOS"] = [None]*len(file_names)

    if cer:
        assert len(file_names)==len(ground_truths), "Number of ground truth transcriptions should be equal to number of files"
        for i in range(len(file_names)):
            results["CER"].append(predict_CER(file_names[i], ground_truths[i], 'openai/whisper-tiny'))
    else:
        results["CER"] = [None]*len(file_names)


    write_results_to_csv(results, output_csv)
    print("Results written to ", output_csv)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run selected speech metrics")
    parser.add_argument("--secs", action="store_true", help="Run predict_SECS (requires reference file)")
    parser.add_argument("--nmos", action="store_true", help="Run predict_nMOS")
    parser.add_argument("--cer", action="store_true", help="Run predict_CER (requires ground truth text)")
    parser.add_argument("--input_csv", type=str, help="Path to the input csv file with format [file_name,ground_truth,ref_file]")
    parser.add_argument("--output_csv", type=str, help="Path to the output result csv file with format [file_name,SECS,nMOS,CER]")

    args = parser.parse_args()

    try:
        result = execute_metrics(args.secs, args.nmos, args.cer, args.input_csv, args.output_csv)
        print(result)
    except ValueError as e:
        print(f"Error: {e}")
