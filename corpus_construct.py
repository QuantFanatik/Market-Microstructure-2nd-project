from multiprocessing import Pool
import pandas as pd
import sys
from collections import Counter
from nltk.corpus import words
import nltk
import numpy as np

# nltk.download('words')
english_vocab = set(words.words())

sys.stdout.write('\rLoading Data...')
df = pd.read_csv('data/cleaned_data_2023.csv')
path_list = df['File'].to_list()
token_list = df['Tokens'].to_list()

def get_vocab(tokens):
    return Counter(eval(tokens))

def get_tf(path_tokens_allowed):
    path, tokens, allowed = path_tokens_allowed
    counter = Counter(token for token in eval(tokens) if token in allowed)
    return (path, {token: counter.get(token, 0) for token in allowed})

vocab = set()
num_docs = len(token_list)
if __name__ == '__main__':
    with Pool() as pool:
        results = []
        token_counts = Counter()
        for idx, token_counter in enumerate(pool.imap(get_vocab, token_list), 1):
            results.append(token_counter)
            sys.stdout.write('\r' + f'Processing tokens {idx}/{num_docs}...')
            sys.stdout.flush()
        
    for token_counter in results:
        token_counts.update(token_counter)

    filtered_token_counts = {
        token.lower(): count
        for token, count in token_counts.items()
        if token.lower() in english_vocab}

    token_counts_df = pd.DataFrame(filtered_token_counts.items(), columns=['Token', 'Count']).sort_values('Count', ascending=False)
    # token_counts_df.to_csv('data/token_counts.csv', index=False)

    print("\nMost common valid English tokens:")
    print(token_counts_df.head(10))

    allowed_tokens = sorted(set(token_counts_df[token_counts_df['Count'] >= 10]['Token']))

    with Pool() as pool:
        results = []
        args = ((path, tokens, allowed_tokens) for path, tokens in zip(path_list, token_list))
        for idx, (path, tf_values) in enumerate(pool.imap(get_tf, args), 1):
            results.append((path, tf_values))
            sys.stdout.write('\r' + f'Computing TF {idx}/{num_docs}...')
            sys.stdout.flush()

    tf_matrix = np.array([tf_values for _, tf_values in results])
    idf_array = np.log(num_docs) - np.log1p(token_counts_df.set_index('Token').loc[allowed_tokens]['Count'].values) 
    tfidf_matrix = tf_matrix * idf_array

    # Create DataFrame for TF-IDF
    tfidf_df = pd.DataFrame(
        data=tfidf_matrix,
        index=[path for path, _ in results],
        columns=allowed_tokens
    ).reset_index()
    tfidf_df.rename(columns={'index': 'File'}, inplace=True)
    tfidf_df.to_csv('data/tfidf_matrix.csv', index=False)

    

