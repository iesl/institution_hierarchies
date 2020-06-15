# Institution Hierarchies # 

[Predicting Institution Hierarchies with Set Based Models](https://openreview.net/pdf?id=pJg1LahGc0)
Derek Tam, Nicholas Monath, Ari Kobren, Andrew McCallum. Automated Knowledge Based Construction (AKBC). 2020.

## Init Setup ##
Setup conda environment: `conda env create --name {env_name} --file=env.yml`

For each session, run `source bin/setup.sh` to set environment variables.

## Install Datasets ## 

The original dataset is at this google drive [link](https://drive.google.com/drive/folders/1wuWQR8RvT6hBRaShZvJaBNy4RR-8rmFM?usp=sharing) 
and the `orig_dataset` directory should be put under the top directory `data`.


## Train Model ## 

First create a config JSON file (sample file at `config/SetAvgEmb/SetAvgEmb_inst.json`).

Then, train the model by running `bin/run/train_model.sh` with the config JSON file as an argument. For example, <br />
`sh bin/run/train_model.sh config/SetAvgEmb/SetAvgEmb_inst.json`

## Evaluating Models ##

For the first option, run `bin/run/test_model.sh`, passing in the experiment directory as the argument. For example, <br />
`sh bin/run/test_model.sh exp_out/SetAvgEmb/inst_city_state_country_type/2020-06-09-10-49-15`. 

## Cross Validation ## 

To install the cross validation datasets, run the following commands: 

`python src/main/setup/init_components.py` <br />
`python src/main/setup/init_cross_validation.py`

To setup cross validation file experiment, setup script to run do grid search over hyperparametres and different folds (assumes slurm manager) 
by running `python src/main/train/setup_grid_search_train.py` over the cross validation config file, passing in the gpu names. For example, <br /> 

`python src/main/train/setup_grid_search_train.py -c config/SetAvgEmb/cv_grid_SetAvgEmb_inst_city_state_country_type.json -g gpu`

To run the best hyperparameter configuration for each fold, run  
`python src/main/eval/cross_validation_eval.py` passing in the exp dir and gpu. 

Finally, to get the average test scores, run `python src/main/eval/get_avg_test.py` passing in the exp dir. 
