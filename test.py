import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cosine_sim_df = pd.read_csv('cosine_similarity.csv', index_col=0)
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_df, cmap='coolwarm', annot=False)
plt.title("Cosine Similarity Between Documents")
plt.show()