import os
import re
import sys
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import tee
from nltk.corpus import words
from collections import Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Important
# Industry selected in line 153
# The least popular tokens are cut in line 233, the rest is automatic.

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
nlp.max_length = 1e9
english_vocab = set(words.words())

def update_progress(index, total, message="Processing file"):
    progress = f"{message} {index}/{total}..."
    sys.stdout.write('\r' + progress)
    sys.stdout.flush()

patterns = {
1:  re.compile(
    r"""
    (?i)i[\s]*t[\s]*e[\s]*m[\s]*\s*1[^\w\s]*\s*                  
    b[\s]*u[\s]*s[\s]*i[\s]*n[\s]*e[\s]*s[\s]*s[\s]*             
    .*?                                                          
    i[\s]*t[\s]*e[\s]*m[\s]*\s*1[^\w\s]*\s*                      
    b[\s]*u[\s]*s[\s]*i[\s]*n[\s]*e[\s]*s[\s]*s[\s]*             
    (.*?)                                                        
    i[\s]*t[\s]*e[\s]*m[\s]*\s*2[^\w\s]*\s*                      
    p[\s]*r[\s]*o[\s]*p[\s]*e[\s]*r[\s]*t[\s]*i[\s]*e[\s]*s[\s]* 
    """,
    re.DOTALL | re.VERBOSE),

2:  re.compile(
    r"""
    (?i)                                
    i[\s]*t[\s]*e[\s]*m[\s]*1[\s]*[.,]? 
    (?![\s]*[AaBb\d])                   
    .*?                                 
    i[\s]*t[\s]*e[\s]*m[\s]*1[\s]*[.,]? 
    (?![\s]*[AaBb\d])                    
    (.*?)                               
    (?=i[\s]*t[\s]*e[\s]*m[\s]*\s*(2[^\w\s]*|[3-9]|[1-9][0-9])) 
    """,
    re.DOTALL | re.VERBOSE),

3:  re.compile(
    r"""
    (?i)i[\s]*t[\s]*e[\s]*m[s]?\s*1[\s.,:] 
    (.{10000,}?)                                                        
    (?=i[\s]*t[\s]*e[\s]*m\s*[2-9]|i[\s]*t[\s]*e[\s]*m\s*[1-9]\d)                                    
    """,
    re.DOTALL | re.VERBOSE),

4:  re.compile(
    r"""
    (?i)d[\s]*e[\s]*s[\s]*c[\s]*r[\s]*i[\s]*p[\s]*t[\s]*i[\s]*o[\s]*n[\s]*  
    o[\s]*f[\s]*                                                       
    b[\s]*u[\s]*s[\s]*i[\s]*n[\s]*e[\s]*s[\s]*s?[\s]*             
    (.{10000,}?)                                                        
    (?=item\s*[2-9]|item\s*[1-9]\d)  
    """,
    re.DOTALL | re.VERBOSE)
}

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text = re.sub(r'[^\S ]', '', text)
            text = re.sub(r'\s+', ' ', text)
            for current_pattern in patterns.values():
                match = current_pattern.search(text)
                if match:
                    matched_text = match.group(1).strip()
                    if len(matched_text) >= 2000:  # Valid match
                        return (file_path, matched_text, None)

            return (file_path, None, "Invalid")
    except Exception as e:
        return (file_path, None, f"Error: {e}")

def process_text(file_path):
    path, text = file_path # TODO: Read the file
    doc = nlp(text)
    filtered_tokens = [
    token.text.lower()
    for token in doc
    if token.is_alpha and not token.is_stop and token.text.lower() in english_vocab]
    return (path, filtered_tokens)

def get_vocab(tokens):
    return Counter(tokens)

def get_tf(path_tokens_allowed):
    path, tokens, allowed = path_tokens_allowed
    counter = Counter(token for token in tokens)
    tf_values = [counter.get(token, 0) for token in allowed]
    return (path, tf_values)

if __name__ == '__main__':
    
    #-------------Classifications----------------#
    base_directory = "data/2023"
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, base_directory)

    unique_lines = set()
    invalid_counter = 0

    paths, idx = tee((os.path.join(root_dir, file)
                    for root_dir, _, files in os.walk(path)
                    for file in files if not file.startswith('.')))

    frame_dict = defaultdict(list)
    total_files = len(list(idx))
    for idx, file_path in enumerate(paths, 1):
        update_progress(idx, total_files, "Fetching Classifications")
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(26):
                f.readline()
            line = f.readline().strip()
            if line.startswith("STANDARD INDUSTRIAL CLASSIFICATION:"):
                clean_line = line.split("\t")[1].strip()
                unique_lines.add(clean_line)
                frame_dict[clean_line].append(os.path.relpath(file_path, root))
            else:
                invalid_counter += 1

                    
    rows = ({"SIC": sic, "File": file} 
            for sic, files in frame_dict.items() 
            for file in files)

    replace_str = {'[3949]': 'SPORTING & ATHLETIC GOODS, NEC [3949]',
                '[6221]': 'COMMODITY CONTRACTS BROKERS & DEALERS [6221]'}

    df = pd.DataFrame(rows).apply(lambda x: x.replace(replace_str), axis=1).sort_values(by='SIC')
    # df.to_csv('data/data_2023.csv', index=False)
    # print(df)

    #---------------Descriptions-----------------#
    df = df.loc[df['SIC'] == 'FIRE, MARINE & CASUALTY INSURANCE [6331]', slice(None)].set_index('File')
    paths = df.index.tolist()
    total_files = len(paths)
    invalid_files = []

    with Pool() as pool:
        results = []
        for idx, result in enumerate(pool.imap(process_file, paths), 1):
            update_progress(idx, total_files, 'Extracting Descriptions')
            results.append(result)

    for file_path, matched_text, error in results:
        if matched_text:
            df.at[file_path, 'Text'] = matched_text
        elif error:
            invalid_files.append(file_path)

    print(invalid_files)
    if not len(invalid_files) == 0:
        with open('invalid_files.txt', 'w') as f:
            f.write('This file contains the list of unmatched or imporoperly matched files.\n')
            for file in set(invalid_files):
                f.write(file + '\n')

    sys.stdout.write('\rProcessing complete! ' + ' ' * 50 + '\n')

    df = df.reset_index().dropna()
    # df.to_csv('data/reduced_data_2023.csv', index=False)
    print(f"Total invalid files: {len(set(invalid_files))}")
    # print(df)

    #---------------Lemmatization-----------------#
    # ATTENTION! This operation is extremely slow and may take hours to complete
    # if executed on the entire dataset. It is best to use a subset of the data.
    # The output of a full execution is available in data/cleaned_data_2023.csv.

    texts = df['Text'].tolist()
    df.drop('Text', axis=1, inplace=True)
    total_files = len(texts)
    texts = [(path, text) for path, text in zip(df.index.to_list(), texts)]
    df['Tokens'] = [[] for _ in range(len(df))]

    with Pool() as pool:
        results = []
        for idx, result in enumerate(pool.imap(process_text, texts), 1):
            update_progress(idx, total_files, 'Lemmatizing Text')
            results.append(result)

    for path, tokens in results:
        try:
            df.at[path, 'Tokens'] = tokens
        except Exception as e:
            print(e)

    sys.stdout.write('\rProcessing complete! ' + ' ' * 50 + '\n')

    df = df.reset_index().dropna()
    # df.to_csv('data/reduced_subset_tokens_2023.csv', index=False)
    # print(df)

    #---------------TF-IDF-----------------#
    path_list = df['File'].to_list()
    token_list = df['Tokens'].to_list()
    with Pool() as pool:
        results = []
        token_counts = Counter()
        for idx, token_counter in enumerate(pool.imap(get_vocab, token_list), 1):
            update_progress(idx, total_files, 'Processing tokens')
            results.append(token_counter)
        
    for token_counter in results:
        token_counts.update(token_counter)

    filtered_token_counts = {token: count for token, count in token_counts.items()}
    token_counts_df = pd.DataFrame(filtered_token_counts.items(), columns=['Token', 'Count']).sort_values('Count', ascending=False)
    token_counts_df.to_csv('data/token_counts.csv', index=False)

    # print("\nMost common valid English tokens:")
    # print(token_counts_df.head(10))

    allowed_tokens = sorted(set(token_counts_df[token_counts_df['Count'] >= 10]['Token']))

    with Pool() as pool:
        results = []
        args = ((path, tokens, allowed_tokens) for path, tokens in zip(path_list, token_list))
        for idx, (path, tf_values) in enumerate(pool.imap(get_tf, args), 1):
            update_progress(idx, total_files, 'Computing TF')
            results.append((path, tf_values))

    tf_matrix = np.array([tf_values for _, tf_values in results])
    idf_array = np.log1p(total_files) - np.log1p(token_counts_df.set_index('Token').loc[allowed_tokens]['Count'].values) 
    tfidf_matrix = tf_matrix * idf_array

    tfidf_df = pd.DataFrame(
        data=tfidf_matrix,
        index=[path for path, _ in results],
        columns=allowed_tokens
    ).reset_index()
    tfidf_df.rename(columns={'index': 'File'}, inplace=True)

    # COSINE SIMILARITY
    tfidf_matrix_normalized = tfidf_matrix / np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)

    cosine_sim = cosine_similarity(tfidf_matrix_normalized)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=[path for path, _ in results],
        columns=[path for path, _ in results]
    )

    cosine_sim_df.to_csv('cosine_similarity.csv', index=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_df, cmap='coolwarm', annot=False)
    plt.title("Cosine Similarity Between Documents")
    plt.show()