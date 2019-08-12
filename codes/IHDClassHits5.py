################################################################################
# reads MUSHv2 emodel files and return Hits@1 for has_class relation
################################################################################

import numpy as np
import pandas as pd
import os
import sys

# SET YOUR TARGET DATASET:
input = "./norange/IHDv2_extracted_norange_withPaths.csv" # only for MUSHv2 now

df = pd.read_csv(input, index_col=0)

output = "./norange/IHDv2_inSocialGroup_hitsAt5_norange.csv"

################################################################################
############################# READ FILES ##############################
################################################################################

resultsTable = pd.DataFrame()

for index, row in df.iterrows():

    filePath = row["filePath"] # eg. "./BPA/model-H1A/BPA..."

    with open(filePath) as f:
        content = f.readlines()
    content = [x.strip() for x in content] # remove white space (needed)

    ### find position of entity and relation embeddings
    scoreLine = [s for s in enumerate(content) if "[score.properties]" in s][0][0]
    entityLine = [s for s in enumerate(content) if "[entity.embeddings]" in s][0][0]

    ### get entity chunk
    scoreChunk = content[scoreLine:entityLine]

################################################################################
################################## SCORE PART ##################################
################################################################################

    ### useful data from scoreChunk
    # filter test sets and overall (*) entries
    hitsAt5 = [set for set in scoreChunk if "score.set.TST.rel.inSocialGroup.tail.hits@5" in set]

    # split chunk of text into distinct values
    hitsAt5 = hitsAt5[0].split(" = ")
    hitsAt5 = ["inSocialGroupHitsAt5"] + [hitsAt5[1]]

    # list of lists to dictionary
    scoreDict = {hitsAt5[0]:[hitsAt5[1]] for score in hitsAt5} # df columns from dict values must be as list

# section output: scoreDict

    summaryTable = pd.DataFrame(scoreDict, index=[index])
    resultsTable = resultsTable.append(summaryTable)

resultsTable.to_csv(output)
