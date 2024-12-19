import os
import re
import sys
import pandas as pd
from itertools import tee
from multiprocessing import Pool
from collections import defaultdict

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

        update_progress(idx, total_files, "Fetching Classifications")
                    
    rows = ({"SIC": sic, "File": file} 
            for sic, files in frame_dict.items() 
            for file in files)

    replace_str = {'[3949]': 'SPORTING & ATHLETIC GOODS, NEC [3949]',
                '[6221]': 'COMMODITY CONTRACTS BROKERS & DEALERS [6221]'}

    df = pd.DataFrame(rows).apply(lambda x: x.replace(replace_str), axis=1).sort_values(by='SIC')
    df.to_csv('data/data_2023.csv', index=False)
    print(df)

    #---------------Descriptions-----------------#
    df = pd.read_csv('data/data_2023.csv', index_col='File')      
    paths = df.index.tolist()
    total_files = len(paths)
    invalid_files = []

    with Pool() as pool:
        results = []
        for idx, result in enumerate(pool.imap(process_file, paths), 1):
            results.append(result)
            update_progress(idx, total_files, 'Extracting Descriptions')

    for file_path, matched_text, error in results:
        if matched_text:
            df.at[file_path, 'Text'] = matched_text
        elif error:
            invalid_files.append(file_path)

    with open('invalid_files.txt', 'w') as f:
        f.write('This file contains the list of unmatched or imporoperly matched files.\n')
        for file in set(invalid_files):
            f.write(file + '\n')

    sys.stdout.write('\rProcessing complete! ' + ' ' * 50 + '\n')

    df = df.reset_index().dropna()
    df.to_csv('data/updated_data_2023.csv', index=False)
    print(f"Total invalid files: {len(set(invalid_files))}")
    print(df)