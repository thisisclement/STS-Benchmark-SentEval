#TODO: Test script if working

import sys
sys.path.insert(0, '..')

from SimilarityEvaluation.similarityevaluation import EmbeddingSimilarityEval_STSB, SimilarityFunction
import os

import argparse

parser = argparse.ArgumentParser(description='Running STS-B benchmark')

parser.add_argument("-d", "--data_path", help="The data path for benchmarking.")
parser.add_argument("-m", "--model_path", help="The model path for benchmarking.")
parser.add_argument("-n", "--benchmark_name", help="Name of Benchmark run.")
parser.add_argument("-o", "--output", help="Output path of the for Benchmark results.")
parser.add_argument("-ssl", help="SSL Cert path (optional)")

args = parser.parse_args()

if args.ssl:
    os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca.xtraman.org.pem' # to solve the SSL cert issue

eval_data_path = args.data_path

if args.model_path:
    model_local_path = args.model_path
    model_list = [args.model_path + "/paraphrase-multilingual-mpnet-base-v2", args.model_path + "/paraphrase-multilingual-MiniLM-L12-v2"]
else:
    model_list = ["paraphrase-multilingual-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]

# running for each model
for model in model_list:
    sts_eval = EmbeddingSimilarityEval_STSB(model, eval_data_path, main_similarity=SimilarityFunction.COSINE, name=args.benchmark_name, show_progress_bar=True, write_csv=True)
    sts_eval.run_eval(output_path=args.output) 