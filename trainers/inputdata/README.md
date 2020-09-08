# Input Data

Note: Sources that are not publicly available or are too big are not included in this repository!

## Named Entity Recognition

Reference papers:

* Ineke Schuurman, Veronique Hoste & Paola Monachesi (2009) - [Cultivating Trees: Adding Several Semantic Layers to the Lassy Treebank in SoNaR ](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.332.8142)
* Bart DeSmet & Veronique Hoste (2013) - [Fine-grained Dutch Named Entity Recognition](https://biblio.ugent.be/publication/4246431/file/5787347.pdf) - Language Resources and Evaluation. 48. http://dx.doi.org/10.1007/s10579-013-9255-y

The original text is from the Lassy Klein corpus.

Note: Data set splits are our own!

Size (wc output):

```
   1767    3334   13994 sonar1-ner-nld.dev.txt
   1667    3134   13135 sonar1-ner-nld.test.txt
1018506 1918734 8028977 sonar1-ner-nld.train.txt
```

Checksums (md5sum):
```
531270c127b1836ecbd6a55ae1bfada3  sonar1-ner-nld.dev.txt
0290a9270f3d90cd55a73b9da72d1012  sonar1-ner-nld.test.txt
28e8a633a056505efcbcdc400d325f53  sonar1-ner-nld.train.txt
```

## Part of Speech Tagging

Based on Frog.mbt.1.1, a corrected version of Frog.mbt.1.1 where the 43 LET() errors have been fixed. A merge of cgn-wotan-dcoi.mbt with lassy-klein

Note: Data set splits are our own!

Size (wc output):
```
     1096      1992     22894 frog1.1-pos-nld.dev.txt
     1148      2096     23886 frog1.1-pos-nld.test.txt
 11775931  21565528 246115172 frog1.1-pos-nld.train.txt
```

Checksums (md5sum):

```
6429be74794f05a79cb3fef7df0412be  frog1.1-pos-nld.dev.txt
bead336dd429484ff7318bd9c87df0cc  frog1.1-pos-nld.test.txt
ff61b7e64c6c1935efd8fbb02d528729  frog1.1-pos-nld.train.txt
```

## Lemmatisation

Based on the CGN corpus and mblem.lex , a corrected version of E-Lex.

Note: Data set splits are our own!
