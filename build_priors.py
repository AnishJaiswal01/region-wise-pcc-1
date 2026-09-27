import pandas as pd
import json

df = pd.read_csv("IHME-GBD_2023_DATA-44b96e82-1.csv")

# Map the broad IHME categories directly to torchxrayvision's exact output strings
disease_mapping = {
    "Tracheal, bronchus, and lung cancer": ["Mass", "Nodule", "Lung Lesion"],
    "Lower respiratory infections": ["Pneumonia", "Consolidation", "Infiltration"],
    "Cardiovascular diseases": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    "Chronic respiratory diseases": ["Emphysema", "Fibrosis", "Pleural_Thickening"]
}

state_priors = {}

for state, group in df.groupby("location_name"):
    state_priors[state] = {}
    
    for _, row in group.iterrows():
        ihme_cause = row["cause_name"]
        prevalence = row["val"]
        
        # Expand the single IHME prevalence rate into the specific AI output nodes
        if ihme_cause in disease_mapping:
            for xrv_pathology in disease_mapping[ihme_cause]:
                state_priors[state][xrv_pathology] = prevalence

with open("state_disease_priors.json", "w") as f:
    json.dump(state_priors, f, indent=4)