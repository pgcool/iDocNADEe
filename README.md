# iDocNADEe 
AAAI 2019 paper: "Document Informed Neural Autoregressive Topic Models with Distributional Prior" 
(a Contextualized Neural Topic Model with Word Embeddings) 

## About
This code consists of the implementations for the proposed models: DocNADE, iDocNADE, DocNADEe and iDocNADEe in the AAAI-19 paper.

## Requirements
Requires Python 3 (tested with `3.6.1`). The remaining dependencies can then be installed via:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('all')"


## Data format

**Datasets**: A directory containing CSV files. There is expected to be 1 CSV file per set or collection, with separate sets for training, validation and test. The CSV files in the directory must be named accordingly: `training_docnade.csv`, `validation_docnade.csv`, `test_docnade.csv`. For this task, each CSV file (prior to preprocessing) consists of 2 string fields with a comma delimiter - the first is the label and the second is the document body.

**Vocabulary files**: A plain text file, with 1 vocabulary token per line (note that this must be created in advance, we do not provide a script for creating vocabularies).


## How to use: Training of ``DocNADE``, ``DocNADEe``, ``iDocNADE`` and ``iDocNADEe``

The script ``train_TMN_docnade_TASKTYPE.sh`` invokes ``train_model.py`` to train the four different model to compute PPL as well IR and save it in a repository. It will also log all the information with the PPL and IR in the same model folder. Here's how to use the script:

        $ ./train_TMN_docnade_PPL.sh
		
		$ ./train_TMN_docnade_IR.sh
		
        - ``dataset`` 				is the path to the input dataset.
        - ``docnadeVocab`` 			is the path to vocabulary file used by DocNADE.
        - ``model`` 				is the path to model output directory.
        - ``learning-rate`` 		is learning rate.
        - ``batch-size`` 			is batch size for training data.
        - ``num-steps`` 			is the number of steps to train for.
        -  `log-every`` 			is to print training loss after this many steps.
        - ``validation-bs`` 		is the batch size for validation evaluation.
        - ``test-bs`` 				is the batch size for test evaluation.
        - ``validation-ppl-freq`` 	is to evaluate validation PPL and NLL after this many steps.
        - ``validation-ir-freq`` 	is to evaluate validation IR after this many steps.
        - ``test-ir-freq`` 			is to evaluate test IR after this many steps.
        - ``test-ppl-freq`` 		is to evaluate test PPL and NLL after this many steps.
        - ``num-classes`` 			is number of classes.
        - ``patience`` 				is patience for early stopping criterion.
        - ``hidden-size`` 			is size of the hidden layer.
        - ``activation`` 			is which activation to use: sigmoid|tanh. Notice, use 'sigmoid' for **PPL** and 'tanh' for **IR** computations. 
        - ``bidirectional`` 		is whether to use iDocNADE model or not,  i.e. True or False. If True, then model --> ``iDocNADE``
		- ``initialize-docnade`` 	is whether to include glove embedding prior or not, i.e. True or False. If True, then model -->``DocNADEe``. 
									If ``bidirectional`` = True and ``initialize-docnade`` = True, then model --> ``iDocNADEe``.
        - ``combination-type`` 		is combination type for iDocNADE forward and backward hidden document representation, for instance 'sum'
        - ``vocab-size`` 			is the vocabulary size.
		- ``lambda-embeddings``		is the mixture weight, i.e. [0.0-1.0] with word embeddings. 
        - ``projection`` 			is whether to project prior embeddings or not,  i.e. True or False. Set to False.
        - ``deep`` 					is whether to maked model deep (deepDocNADE) or not (docNADE/iDocNADE),  i.e. True or False
        - ``deep-hidden-sizes`` 	is sizes of the deep hidden layers for deepDocNADE, for instance, 200, 200.
        - ``reload`` 				is whether to reload model or not,  i.e. True or False
        - ``reload-model-dir`` 		is path of directory for which model to be reloaded.
        - ``trainfile`` 			is path to training text file (required in case of topic coherence), for instance, ./datasets/20NSshort/training.txt
        - ``valfile`` 				is path to validation text file. (required in case of topic coherence), for instance, ./datasets/20NSshort/validation.txt
        - ``testfile`` 				is path to test text file. (required in case of topic coherence), for instance, ./datasets/20NSshort/test.txt
		- ``pretrained-embeddings-path`` is the path to pre-trained DocNADE model to initialize iDocNADE/DocNADEe/iDocNADEe model for **PPL** computation, for instance, './docnade_embeddings_ppl_reduced_vocab/TMNtitle'


# Directory structure for results and datasets


## Contains dataset folders
Datasets directory:             ./datasets/

## Contains GloVe pretrained embeddings
Pre-trained embeddings dir:     /home/usr/resources/pretrained_embeddings/

## Contains results of training
Results directory:              ./model/MODELNAME/

Saved PPL model dir:            ./model/MODELNAME/model_ppl/

Saved IR model dir:             ./model/MODELNAME/model_ir/


Saved logs model dir:           ./model/MODELNAME/logs/

Reload IR results:              ./model/MODELNAME/logs/reload_info_ir.txt

Reload PPL results:             ./model/MODELNAME/logs/reload_info_ppl.txt


## Reload Functionality: In case of reload, set the following:
--reload-model-dir             
--trainfile                   
--valfile                    
--testfile     

NOTE: In computing PPL or IR for larget text datasets, it is recommented to use the reload functionality for generating scores (PPL/IR) for the test set, and set ``test-ppl-freq`` or ``test-ir-freq`` to a very large number so as to avoid overhead during the training time.

# Citation

@inproceedings{pankajgupta2018iDocNADEe,
title={Document Informed Neural Autoregressive Topic Models with Distributional Prior},
author={Gupta, Pankaj and Chaudhary, Yatin and Buettner, Florian and Sch{\"u}tze, Hinrich},
booktitle={AAAI},
year={2018}
}
