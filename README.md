<div align="center">
  <img src="https://raw.githubusercontent.com/proycon/deepfrog/master/logo.png" width="200" />
</div>

# DeepFrog - NLP Suite

[![Language Machines Badge](http://applejack.science.ru.nl/lamabadge.php/deepfrog)](http://applejack.science.ru.nl/languagemachines/)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


## Introduction

DeepFrog aims to be a (partial) successor of the Dutch-NLP suite [Frog](https://languagemachines.github.io/frog).
Whereas the various NLP modules in Frog wre built on k-NN classifiers, DeepFrog builds on deep learning techniques and
can use a variety of neural transformers.

Our deliverables are multi-faceted:

1. Fine-tuned neural network **models** for Dutch NLP that can be compared with [Frog](https://languagemachines.github.io/frog) and are
    directly usable with [Huggingface's Transformers](https://github.com/huggingface/transformers) library for Python (or
    [rust-bert](https://github.com/guillaume-be/rust-bert) for Rust).
2. **Training pipelines** for the above models (see ``training``).
3. A **software tool** that integrates multiple models (not just limited to dutch!) and provides a single pipeline solution for end-users.
    * with full support for [FoLiA XML](https://proycon.github.io/folia) input/output.
    * usage is not limited to the models we provide

## Models

We aim to make available various models for Dutch NLP.

#### RobBERT v1 Part-of-Speech (CGN tagset) for Dutch

Model page with instructions: https://huggingface.co/proycon/robbert-pos-cased-deepfrog-nld

Uses pre-trained model [RobBERT](https://people.cs.kuleuven.be/~pieter.delobelle/robbert/) (a Roberta model), fine-tuned on
part-of-speech tags with the full corpus as also used by Frog. Uses the tag set of Corpus Gesproken Nederlands (CGN), this
corpus constitutes a subset of the training data.

Test Evaluation:

```
f1 = 0.9708171206225681
loss = 0.07882563415198372
precision = 0.9708171206225681
recall = 0.9708171206225681
```

#### RobBERT v2 Part-of-Speech (CGN tagset) for Dutch

Uses pre-trained model [RobBERT v2](https://people.cs.kuleuven.be/~pieter.delobelle/robbert/) (a Roberta model), fine-tuned on
part-of-speech tags with the full corpus as also used by Frog. Uses the tag set of Corpus Gesproken Nederlands (CGN), this
corpus constitutes a subset of the training data.

```
f1 = 0.9801121876593575
loss = 0.06283498580495898
precision = 0.9796126401630989
recall = 0.9806122448979592
```

#### BERT Part-of-Speech (CGN tagset) for Dutch

Model page with instructions: https://huggingface.co/proycon/bert-pos-cased-deepfrog-nld

Uses pre-trained model [BERTje](https://github.com/wietsedv/bertje) (a BERT model), fine-tuned on
part-of-speech tags with the full corpus as also used by Frog. Uses the tag set of Corpus Gesproken Nederlands (CGN), this
corpus constitutes a subset of the training data.

Test Evaluation:

```
f1 = 0.9737354085603113
loss = 0.0647074995296342
precision = 0.9737354085603113
recall = 0.9737354085603113
```

#### RobBERT SoNaR1 Named Entities for Dutch

Model page with instructions: https://huggingface.co/proycon/robbert-ner-cased-sonar1-nld

Uses pre-trained model [RobBERT](https://people.cs.kuleuven.be/~pieter.delobelle/robbert/) (a Roberta model), fine-tuned on
Named Entities from the SoNaR1 corpus (as also used by Frog). Provides basic PER,LOC,ORG,PRO,EVE,MISC tags.


Test Evaluation (**note:** *this is a simple token-based evaluation rather than entity based!*)

```
f1 = 0.9170731707317074
loss = 0.023864904676364467
precision = 0.9306930693069307
recall = 0.9038461538461539
```

**Note:** the tokenisation in this model is English rather than Dutch

#### BERT SoNaR1 Named Entities for Dutch

Model page with instructions: https://huggingface.co/proycon/bert-ner-cased-sonar1-nld

Uses pre-trained model [BERTje](https://github.com/wietsedv/bertje) (a BERT model), fine-tuned on
Named Entities from the SoNaR1 corpus (as also used by Frog). Provides basic PER,LOC,ORG,PRO,EVE,MISC tags.

Test Evaluation (**note:** *this is a simple token-based evaluation rather than entity based!*)

```
f1 = 0.9519230769230769
loss = 0.02323892477299803
precision = 0.9519230769230769
recall = 0.9519230769230769
```
