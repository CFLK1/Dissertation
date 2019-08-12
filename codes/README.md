## Datasets
The datasets are referred to a tentative name in the codes, which is different from their presented names:\
IHDv2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &nbsp;&nbsp;Demographics\
IHPv2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &nbsp;&nbsp;Pregnancy\
MUSHv2 &nbsp;&nbsp;= &nbsp;&nbsp;Mushroom\
PTREEv2 = &nbsp;&nbsp;BPA-Tiny\
PROC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &nbsp;&nbsp;BPA-Small\
N2N &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &nbsp;&nbsp;BPA-Medium\
FULL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= &nbsp;&nbsp;BPA-Large

## Extract model data
The output of embedding models are in .emodel files. They are separated into folders by dataset and stored in\
the working directory.

"emodel2df.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: all .emodel files in a dataset folder\
&nbsp;&nbsp;&nbsp;&nbsp; output: one "DATASET_extracted_withPaths.csv" for each dataset, containing hyperparameters, scores and the corresponding file paths of each model.

## Classification
#### Labelled data
Based on an excel file of randomly train-val-test splitted labelled data for each dataset's specific task, a "DATASET_labels.csv" is constructed for each dataset.

#### Extract particular entity embedding vectors for classifier
"embed4ML_DATASET.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_labels.csv", "DATASET_extracted_withPaths.csv", .emodel files\
&nbsp;&nbsp;&nbsp;&nbsp; output: For each model in "extracted_withPaths.csv", the entity vectors are retrieved from the .emodel file and combined with the labels in a csv stored in "./classification_data" by dataset.

"MLP_DATASET.py":\
Runs MLP, this can take from 10 mins to an hour for each model.\
&nbsp;&nbsp;&nbsp;&nbsp; input: labelled embedding vectors .csv files\
&nbsp;&nbsp;&nbsp;&nbsp; output: the best model from callback for each model is stored in a best_dir directory eg. "./MLP_model"; the scores are combined with the model settings and streamed into a single "DATASET_MLP.csv".

#### Combine classification scores and emodel data
"DATASETClassHitsN.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_extracted_withPaths.csv", .emodel files\
&nbsp;&nbsp;&nbsp;&nbsp; output: According to the classification task of the dataset, the corresponding Hits@1 or Hits@5 for that particular relation is retrieved and returned in a single "DATASETClassHitsN.csv" file.

"join_MLP_data.py":\
Run for each dataset.\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_extracted_withPaths.csv", "DATASET_MLP.csv", "DATASETClassHitsN.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: Join the tables by matching model hyperparameter settings, returning the final "DATASET_jointable.csv"

## Heatmaps / colourtables
"colourtable.py":\
File contains an adjustable loop over all datasets and metrics (not MR, too sparse).
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_jointable.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_METRIC_colourtab.png"

## Top50 (and Top100 but not used)
"top.py":\
Run for each dataset and change the metric.\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_jointable.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_Top50_METRIC.csv", "DATASET_Top100_METRIC.csv"

## Random Forest regressor + partial dependency plots
"RF_regress.py", "RF_regress_bydim" (by dimension):\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_jointable.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: in a dedicated directory, opens a subdirectory for each dataset, opens a subsubdirectory for each metric as learning target, returns feature importance plots and partial dependence plots.

## Spearman's correlation and scatterplots
"spearman_DATASET.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_jointable.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_spearmanr.csv"

"scatterplot.py":\
Choose X and Y.\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_jointable.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_X_Y_scatterplot.png"

## Counting best options when holding others constant (not used)
"queryv2.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_extracted_withPaths.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_hyperparameters.csv"

"modelcompare.py":\
&nbsp;&nbsp;&nbsp;&nbsp; input: "DATASET_extracted_withPaths.csv"\
&nbsp;&nbsp;&nbsp;&nbsp; output: "DATASET_stats.csv"
