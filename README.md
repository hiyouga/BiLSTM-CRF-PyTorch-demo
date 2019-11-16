# BiLSTM-CRF-PyTorch-demo

> A simple baseline model for Named Entity Recognition.

## Requirement

- PyTorch >= 0.4.0
- NumPy >= 1.13.3
- Python 3.6
- GloVe pre-trained word vectors:
  * Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  * Extract the [glove.42B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) to the `\glove\` folder.

## Dataset

Based on the shared task of Named Entity Recognition on CoNLL 2003 (English corpus). [[link]](https://www.clips.uantwerpen.be/conll2003/ner)

## Train

```sh
python train.py
```

## Reference

- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. "Neural Architectures for Named Entity Recognition." Proceedings of NAACL-HLT (pp. 260-270). 2016. [[pdf]](https://www.aclweb.org/anthology/N16-1030.pdf)

- Linear-chain conditional random field implemented by kmkurn: [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

## License

MIT