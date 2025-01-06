from multiprocessing import Pool
import pandas as pd
import sys
from collections import Counter
from nltk.corpus import words
import nltk

sys.stdout.write('\rLoading Data...')

token_df = pd.read_csv('data/token_counts.csv')
file_list = pd.read_csv('data/cleaned_data_2023.csv', usecols=['File'])

tf_df = pd.DataFrame(index=file_list['File'], columns=token_df['Token'], data=0)
print(tf_df)

def get_tf

# vocab = set()
# total = len(token_list)
# if __name__ == '__main__':
#     with Pool() as pool:
#         results = []
#         token_counts = Counter()
#         for idx, token_counter in enumerate(pool.imap(get_vocab, token_list), 1):
#             results.append(token_counter)
#             sys.stdout.write('\r' + f'Processing tokens {idx}/{total}...')
#             sys.stdout.flush()
        
#     for token_counter in results:
#         token_counts.update(token_counter)

#     filtered_token_counts = {
#         token.lower(): count
#         for token, count in token_counts.items()
#         if token.lower() in english_vocab}

#     filtered_counts_df = pd.DataFrame(filtered_token_counts.items(), columns=['Token', 'Count']).sort_values('Count', ascending=False)
#     filtered_counts_df.to_csv('data/token_counts.csv', index=False)

#     print("\nMost common valid English tokens:")
#     print(filtered_counts_df.head(10))