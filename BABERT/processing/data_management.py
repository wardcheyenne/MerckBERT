import config
import pandas as pd
df = pd.read_csv(config.TRAINING_PATH)

# Split labels based on whitespace and turn them into a list
labels = [i.split() for i in df['labels'].values.tolist()]

# Check how many labels are there in the dataset
unique_labels = set()

for lb in labels:
  [unique_labels.add(i) for i in lb if i not in unique_labels]
 
#print(unique_labels)

# Map each label into its id representation and vice versa
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
#print(labels_to_ids)
