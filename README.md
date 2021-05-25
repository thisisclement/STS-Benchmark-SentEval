## STS Benchmark Evaluator 
STS Benchmark Evaluator is a helper library that evaluates [Sentence Transformer](https://github.com/UKPLab/sentence-transformers) models for Semantic Textual Similarity Tasks. 

This utilises the STS-Benchmark test set for the evaluation. This should work for the other SemEval datasets as well. 

### How to run
1. Install dependencies and needed libraries
```bash
pipenv install  
```
1. You can take a look at my example notebook in `run_stsbenchmark.ipynb`

## Further work
- Increase test scope to STS-Benchmark dev set
- Increase test scope to other STS related evaluation (depending on need)

## Citation 

```
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

