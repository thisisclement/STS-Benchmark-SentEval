from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sentence_transformers import SentenceTransformer
from typing import List
from enum import Enum
import numpy as np
import csv
import os


class SimilarityFunction(Enum):
    """
    Similarity functions that are supported.
    """
    
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3

class STSBenchmarkReader:
    """
    STS Benchmark reader to prep the data for evaluation.
    """

    def __init__(self, data_path: str = None):
        assert data_path != None and os.path.isfile(data_path)
        self.data_path = data_path
        data_dict = dict(sent1=[], sent2=[], scores=[])

        with open(data_path) as fopen:
            dataset = list(filter(None, fopen.read().split('\n')))

        sent1 = []
        sent2 = []
        scores = []

        for data in dataset:
            data_list = data.split('\t')
            sent1.append(data_list[5])
            sent2.append(data_list[6])
            scores.append(data_list[4])

        data_dict['sent1'] = sent1
        data_dict['sent2'] = sent2
        data_dict['scores'] = scores
        # sanity check
        assert len(data_dict['sent1']) == len(data_dict['sent2'])
        assert len(data_dict['sent1']) == len(data_dict['scores'])

        self.data = data_dict

class EmbeddingSimilarityEval_STSB:
    """
    Class to compute embeddings, find pair-wise similarity and do model evaluation based on the recommended STS Benchmark test set. 
    """
    def __init__(self, model_path_or_str: str, eval_data_path: str, batch_size: int = 16, main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.
        
        :param models: Model that you want to evaluate with
        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        assert model_path_or_str != None or model_path_or_str != ''
        assert eval_data_path != None or eval_data_path != ''
        
        stsb = STSBenchmarkReader(eval_data_path)
        self.eval_data_path = eval_data_path

        self.model = SentenceTransformer(model_path_or_str)
        if isinstance(model_path_or_str, str) and (model_path_or_str.find('\\') == -1 or model_path_or_str.find('/') == -1):
            self.model_name = model_path_or_str
        elif os.path.isdir(model_path_or_str) and not model_path_or_str.startswith('http://') and not model_path_or_str.startswith('https://'):
            self.model_name = model_path_or_str.split('\\')[-1]
        self.sentences1 = stsb.data['sent1']
        self.sentences2 = stsb.data['sent2']
        self.scores = [float(i) for i in stsb.data['scores']]
        self.write_csv = write_csv
        self.main_similarity = main_similarity
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["model", "stsb_dataset_name", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]
        
        
    def encode_embeddings(self):
        all_sent = list()
        #note down the sent1 end index
        sent1_end_idx = len(self.sentences1)
        #join both sent1 and sent2 into the same list
        all_sent.extend(self.sentences1)
        all_sent.extend(self.sentences2)
        self.sentences = all_sent
        embeddings = self.model.encode(self.sentences, convert_to_numpy=True, show_progress_bar=self.show_progress_bar)
        return embeddings[:sent1_end_idx], embeddings[sent1_end_idx:]
   

    def run_eval(self, output_path: str = None):
        assert self.model_name != None
        embeddings1, embeddings2 = self.encode_embeddings()
        labels = self.scores
        eval_cosine = dict()
        eval_manhattan = dict()
        eval_euclidean = dict()
        eval_dot = dict()
        
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
        
        eval_cosine['pearson'], _ = pearsonr(labels, cosine_scores)
        eval_cosine['spearman'], _ = spearmanr(labels, cosine_scores)
        
        eval_manhattan['pearson'], _ = pearsonr(labels, manhattan_distances)
        eval_manhattan['spearman'], _ = spearmanr(labels, manhattan_distances)

        eval_euclidean['pearson'], _ = pearsonr(labels, euclidean_distances)
        eval_euclidean['spearman'], _ = spearmanr(labels, euclidean_distances)

        eval_dot['pearson'], _ = pearsonr(labels, dot_products)
        eval_dot['spearman'], _ = spearmanr(labels, dot_products)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                    
                writer.writerow([self.model_name, self.eval_data_path, eval_cosine['pearson'], eval_cosine['spearman'], eval_euclidean['pearson'],
                                 eval_euclidean['spearman'], eval_manhattan['pearson'], eval_manhattan['spearman'], eval_dot['pearson'], eval_dot['spearman']])


        if self.main_similarity == SimilarityFunction.COSINE:
            print("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                eval_cosine['pearson'], eval_cosine['spearman']))
            return eval_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_dot
        elif self.main_similarity is None:
            return max(eval_cosine, eval_manhattan, eval_euclidean, eval_dot)
        else:
            raise ValueError("Unknown main_similarity value")
    
    
