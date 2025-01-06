import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Load token counts
df = pd.read_csv('data/token_counts.csv')

# Sort by Count
df = df.sort_values('Count', ascending=False)

# Compute CDF
df['Cumulative'] = df['Count'].cumsum()
df['Cumulative Percentage'] = df['Cumulative'] / df['Count'].sum() * 100

# Plot CDF
plt.figure(figsize=(8, 6))
plt.plot(df['Count'].index, df['Cumulative Percentage'], label='CDF', color='blue')
plt.title('Cumulative Distribution Function (CDF) of Token Frequency')
plt.xlabel('Token Rank (Sorted by Frequency)')
plt.ylabel('Cumulative Percentage (%)')
plt.grid()
plt.legend()
plt.show()


# Create a word cloud
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(df['Token'], df['Count'])))

# Display the word cloud in higher resolution
plt.figure(figsize=(4, 2), dpi=300)  # Increase figsize and dpi for higher resolution
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()