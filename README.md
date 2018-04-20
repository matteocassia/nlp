# Natural Language Processing 
This repository contains a set of tools developed in Python to perform tasks in the domain of natural language processing, such as text summarisation and sentiment classification.
## Getting Started
In order to use the code in this repository, some actions must be completed beforehand.
### Repository Structure
The repository contains three main folders:
* `nlp/` contains the main source code `.py` files
* `test/` contains the unit tests `.py` files for the source code
* `documents/` contains the raw dataset files in various formats
### Prerequisites and Installing
This project was developed with Python 3.6; consequently, this version of the programming language needs to be installed. It can either be downloaded from the [official website](https://www.python.org/getit/) or through:
```
$ sudo apt-get install python3 pip3
```
Installing Python's own package manager will facilitate installing third party packages, such as `numpy`:
```
$ sudo pip3 install numpy
```
To clone this repository:
```
$ git clone git@github.com:matteocassia/nlp.git
```
Most command-line functionalities in this repository expect the current directory to be the root of this project, so it is advised to navigate into it.
```
$ cd nlp
```
## Using the Code
This section covers the basics of how to use and test the code in this repository. A set of command-line tools have been developed to perform the tasks of this project.
### Registering a Dataset
In order to add a dataset to be used for any task, it needs to be registered. This is done by firstly moving the file(s) of the dataset into the `documents/` folder. Next, in the `Dataset` class in the `nlp.dataset` module, a static method needs to be added for parsing the dataset into a `Dataset` object; the function `get_bbc_dataset` is an example of that. Eventually, an entry in the series of `if` statements in the `get_dataset` function needs to be added, specifying a registration name for this dataset. This method is called by the command-line tools, so the name specified here is the name required as a command-line argument. Four datasets are registered at the time of writing: `musical_instruments`, `automotive`, `instant_video` and `bbc`. The first three are product reviews from Amazon, the last contains news articles by the BBC.

### Classification with Cross-Validation
To perform k-fold cross-validation, navigate to the root of the project and use the command:
```
$ python3 -m nlp.cross_validation <folds> <dataset> <classifier> [<classifier_options>]
```
where `folds` is the integer for the number of folds, `dataset` is the name used to register the required dataset and `classifier` is one of `naive_bayes`, `knn` and `id3`. Should the classifier be KNN, an additional argument is required for the number of neighbours.

For example, to perform 10-fold cross-validation on the automotive dataset with the Naïve Bayes classifier, the command would be:
```
$ python3 -m nlp.cross_validation 10 automotive naive_bayes
```
To perform 3-fold cross-validation on the `musical_instruments` dataset with KNN for 7 neighbours, the command would be:
```
$ python3 -m nlp.cross_validation 3 musical_instruments knn 7
```
### Summarisation with Latent Semantic Analysis
To extract the `k` top keywords for the strongest `n` topics in a dataset, the command is:
```
$ python3 -m nlp.latent_semantic_analysis <dataset> <k> <n>
```
For instance, to extract the top 20 keywords for the first 10 topics in the BBC dataset, the command would be:
```
$ python3 -m nlp.latent_semantic_analysis bbc 20 10
```
### Executing the Unit Tests
In order to execute the all the unit tests designed to determine the integrity of the code, navigate to the root of this project folder and execute the following command:
```
$ python3 -m unittest discover test
```
To execute one specific test at a time, use the command:
```
$ python3 -m unittest test.<name_of_the_test>
```
