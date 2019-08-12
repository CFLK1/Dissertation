################################################################################
# reads emodel files of a dataset, returns a table of model configurations,
# convergence, overall (*) scores, and off-boundary embeddings (1, 0.95, 0.9)
################################################################################

import numpy as np
import pandas as pd
import os
#import sys

# SET YOUR TARGET DATASET:
def emodel2df(dataset):

################################################################################
############################## CONFIGURATION PART ##############################
################################################################################

    configData = []

    datasetPath = "/".join((".", dataset)) # eg. "./BPA"
    models = sorted(os.listdir(datasetPath)) # ["models-H1A", "models-H1B"...]

    for model in models:

        # get model type (H1-4, A/B)
        modelType = model.split("-")[1] # ["H1A"]

        # get list of emodel files
        modelPath = "/".join((datasetPath, model)) # eg. "./BPA/models-H1A"
        files = sorted(os.listdir(modelPath)) # ["BPA...D0008...", "BPA...D0016...", ...]

        # split filenames into elements
        modelConfig = []
        if dataset == "MUSHv2": # write exception (add "FILLER") for dataset with shorter name
            for file in files:
                modelConfig.append(file.split("-")) # [..., ["MUSHROOM", ...]]
                modelConfig[-1].insert(1, "FILLER") # [..., ["MUSHROOM", "FILLER", ...]]
                modelConfig[-1].append("/".join((modelPath, file))) # [..., ["MUSHROOM", "FILLER", ..., "./MUSH/model-H1A/MUS..."]]
        else:
            for file in files:
                modelConfig.append(file.split("-")) # [..., ["BPA", "PTREE", ...]]
                modelConfig[-1].append("/".join((modelPath, file))) # [..., ["BPA", "PTREE", ..., "./BPA/model-H1A/BPA..."]]

        # insert model type as the first item of sublists
        for config in modelConfig:
            config.insert(0, modelType[2]) # [["A", "BPA", ...], ...]
            config.insert(0, modelType[0:2]) # [["H1", "A", "BPA", ...], ...]

        #collect file metadata
        configData += modelConfig

    ### clean up metadata and output csv
    # ONLY vs non-ONLY: insert "NA" for non-ONLY, so they're of the same length
    for model in configData:
        if model[4] != "ONLY":
            model.insert(4, "NA")

    # Now that they're at the same length...
    for model in configData:
        # projection vs linear
        model[5] = model[5].split("_")[1].lower()
        # dimension number
        model[6] = model[6][1:]
        # learning factor
        model[7] = model[7][2:]
        # margin
        model[8] = model[8][2:]
        # constraint
        model[9] = model[9].split("_")[0].lower()
        # L0, L1, L2
        model[10] = model[10].split(".")[0]

    # section output: configTable dataframe (12 columns)
    configHeader = ["HType", "ABType", "dataset", "dataName", "only", "LvP", "dimension", "learnFac", "margin", "constr", "LType", "filePath"]
    configTable = pd.DataFrame(configData, columns=configHeader)
    configTable = configTable.astype(dtype={"dimension":"int64", "learnFac":"int64", "margin":"float64"})

################################################################################
############################# LEARNING CYCLE PART ##############################
################################################################################

    resultsTable = pd.DataFrame()

    for index, row in configTable.iterrows():

        filePath = row["filePath"] # eg. "./BPA/model-H1A/BPA..."

        with open(filePath) as f:
            content = f.readlines()
        content = [x.strip() for x in content] # remove white space (needed)

        ### find position of entity and relation embeddings
        scoreLine = [s for s in enumerate(content) if "[score.properties]" in s][0][0]
        entityLine = [s for s in enumerate(content) if "[entity.embeddings]" in s][0][0]
        relationLine = [s for s in enumerate(content) if "[relation.embeddings]" in s][0][0]

        ### split list into chunks
        modelChunk = content[:scoreLine]
        scoreChunk = content[scoreLine:entityLine]
        entityChunk = content[entityLine:relationLine]
        relationChunk = content[relationLine:]

        ############################################################################
        ### useful data from modelChunk
        cycleProp = [prop for prop in modelChunk if "current_learn_cycle" in prop]
        cycle = cycleProp[0].split(" = ")[1]

    # section output: cycle value

################################################################################
################################## SCORE PART ##################################
################################################################################

        ### useful data from scoreChunk
        # filter test sets and overall (*) entries
        testSet = sorted([set for set in scoreChunk if "score.set.TST.rel.*" in set])

        # split chunk of text into distinct values
        scoreSplitted = map(lambda set: set.split(" = "), testSet)
        scoreSplitAgain = map(lambda set: [set[0].split(".")[-1]] + [set[1]], scoreSplitted)

        # list of lists to dictionary
        scoreDict = {score[0]:[score[1]] for score in scoreSplitAgain} # df columns from dict values must be as list

    # section output: scoreDict

################################################################################
######################### EMBEDDING OFF-BOUNDARY PART ##########################
################################################################################

        ### useful data from entityChunk
        # split chunk of text into distinct values
        entitySplitted = map(lambda embed: embed.split("\t"), entityChunk[1:])
        # make table
        entityTable = pd.DataFrame(entitySplitted) # one way to remove first column (relation name)

        embeddingArray = entityTable.iloc[:,1:].astype("float64").values.flatten()
        prop1 = np.mean((embeddingArray < -1) | (embeddingArray > 1))
        prop2 = np.mean((embeddingArray < -0.95) | (embeddingArray > 0.95))
        prop3 = np.mean((embeddingArray < -0.9) | (embeddingArray > 0.9))

    # section output: prop1, prop2, prop3

################################################################################
############################## DATA ASSEMBLE PART ##############################
################################################################################

        # DataFrame for each model
        summaryTable = pd.DataFrame(scoreDict, index=[index])
        summaryTable.insert(loc=0, column="convergeCycle", value=cycle)
        summaryTable["off1"] = prop1
        summaryTable["off95"] = prop2
        summaryTable["off9"] = prop3

        # append individual output to results table
        resultsTable = resultsTable.append(summaryTable)

    # Final table, combined with configs, matched by index
    #configTableClean = configTable.drop(columns=["dataset", "dataName", "filePath"])
    configTableClean = configTable.drop(columns=["dataset", "dataName"])

    finalTable = pd.concat([configTableClean, resultsTable], axis=1, sort=False)
    finalTable.rename(columns={"hits@1":"hitsAt1", "hits@2":"hitsAt2", "hits@3":"hitsAt3", "hits@5":"hitsAt5", "hits@X":"hitsAtX"}, inplace=True)

    # output csv
    fileName = dataset + "_extracted_withPaths.csv"
    finalTable.to_csv(fileName)

################################################################################

# dataset:         dataset name in {FULL, N2N, PROC, PTREEv2, IHDv2, IHPv2, MUSHv2}

emodel2df(dataset="FULL")
