from similarityevaluation import EmbeddingSimilarityEval_STSB, SimilarityFunction
import os

os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca.xtraman.org.pem' # to solve the SSL cert issue

eval_data_path = 'data/sts-test-ml.csv'

model_local_path = os.getenv("MODEL_LOCAL_PATH")

model_list = [model_local_path + "paraphrase-multilingual-mpnet-base-v2", model_local_path + "paraphrase-multilingual-MiniLM-L12-v2"]

for model in model_list:
    sts_eval = EmbeddingSimilarityEval_STSB(model, eval_data_path, main_similarity=SimilarityFunction.COSINE, name="malay-stsb", show_progress_bar=True, write_csv=True)
    sts_eval.run_eval(output_path='./benchmark_results/')