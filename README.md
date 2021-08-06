## STS Benchmark Evaluator 
STS Benchmark Evaluator is a helper library that evaluates [Sentence Transformer](https://github.com/UKPLab/sentence-transformers) models for Semantic Textual Similarity Tasks. 

This utilises the STS-Benchmark test set for the evaluation. This should work for the other SemEval datasets as well. 

### How to run
- Install dependencies and needed libraries
```bash
pipenv install  
```
- You can take a look at my example notebook in `run_stsbenchmark.ipynb`


### Benchmark Results
|Model                                |Dataset                |Spearman STS-B Score|
|-------------------------------------|-----------------------|--------------------|
|paraphrase-multilingual-MiniLM-L12-v2|Tagalog STS-B|0.3394533385140273  |
|paraphrase-multilingual-mpnet-base-v2|Tagalog STS-B|0.36567875871165906 |
|paraphrase-multilingual-MiniLM-L12-v2|Thai STS-B|0.6000013617198022  |
|paraphrase-multilingual-MiniLM-L12-v2|Chinese STS-B|0.6032880514028351  |
|paraphrase-multilingual-MiniLM-L12-v2|Vietnamese STS-B|0.6037541386963938  |
|paraphrase-multilingual-mpnet-base-v2|Chinese STS-B|0.6052726127430685  |
|paraphrase-multilingual-MiniLM-L12-v2|Malay STS-B   |0.6118353309379856  |
|paraphrase-multilingual-mpnet-base-v2|Thai STS-B|0.6192682373584009  |
|paraphrase-multilingual-mpnet-base-v2|Vietnamese STS-B|0.6265472133925664  |
|paraphrase-multilingual-mpnet-base-v2|Malay STS-B   |0.6341583902856095  |

The multilingual models perform relatively well on Bahasa Malay, Vietnamese, Thai and Chinese but performs poorly on Tagalog related STS tasks. This might be due to the lack of parallel data for the Tagalog language. 

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

