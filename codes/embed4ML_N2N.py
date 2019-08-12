################################################################################
# reads emodel files and return a data frame each of entity embedding vectors
################################################################################

import numpy as np
import pandas as pd
import os
import sys

# SET YOUR TARGET DATASET:
input = "./norange/N2N_extracted_norange_withPaths.csv" # only for MUSHv2 now

df = pd.read_csv(input, index_col=0)

# excel for labelled data:
labels = "./norange/PROC_labels_trunc.csv"

label_df = pd.read_csv(labels)
label_df = label_df.sort_values("sample")

################################################################################
############################# READ FILES ##############################
################################################################################

for index, row in df.iterrows():

    filePath = row["filePath"] # eg. "./BPA/model-H1A/BPA..."

    with open(filePath) as f:
        content = f.readlines()
    content = [x.strip() for x in content] # remove white space (needed)

    ### find position of entity and relation embeddings
    entityLine = [s for s in enumerate(content) if "[entity.embeddings]" in s][0][0]
    relationLine = [s for s in enumerate(content) if "[relation.embeddings]" in s][0][0]

    ### get entity chunk
    entityChunk = content[entityLine:relationLine]

################################################################################
######################### SAMPLE EMBEDDINGS ##########################
################################################################################

    ### useful data from entityChunk
    # split chunk of text into distinct values
    entitySplitted = map(lambda embed: embed.split("\t"), entityChunk[1:])
    # make table
    entityTable = pd.DataFrame(entitySplitted) # one way to remove first column (relation name)

    sampleTable = entityTable[entityTable.iloc[:,0].str.contains("Proc:")]
    for index2, row2 in sampleTable.iterrows():
        row2[0] = row2[0].split(":")[1]

    # construct feature names
    features = list(range(1, len(sampleTable.columns)))
    for i, feature in enumerate(features):
        features[i] = "embed" + str(feature)
    colnames = ["sample"] + features

    sampleTable.columns = colnames

    # sort samples
    sampleTable = sampleTable.sort_values("sample") # avoid inplace copy issues
    sampleTable = sampleTable.reset_index(drop=True) # avoid inplace copy issues

    # cast types
    for col in sampleTable[1:]:
        sampleTable[col] = sampleTable[col].astype(dtype="float64")
    sampleTable["sample"] = sampleTable["sample"].astype(dtype="int64")


################################################################################
############################## GET CLASS ##############################
################################################################################

    labelledTable = sampleTable.join(label_df.set_index("sample"), on="sample", how="inner")

    # output csv
    outDir = "/".join([".", "norange", "classification_data", filePath.split("/")[1], filePath.split("/")[2]])
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    outPath = outDir + "/" + filePath.split("/")[3][:-7] + ".csv"
    labelledTable.to_csv(outPath)
