import pandas as pd
import selfies as sf
import torch
import numpy as np

def prepare_data():
    # load the data
    df = pd.read_csv('data/zinc_subset.csv')
    
    # remove any hidden newlines or spaces from the SMILES column
    df['smiles'] = df['smiles'].str.strip()
    
    df['selfies'] = df['smiles'].apply(sf.encoder)
    
    # build vocabs
    all_smiles_chars = sorted(list(set("".join(df['smiles']))))
    smiles_vocab = {char: i+1 for i, char in enumerate(all_smiles_chars)}
    smiles_vocab["[PAD]"] = 0 
    
    all_selfies_tokens = sorted(list(sf.get_alphabet_from_selfies(df['selfies'])))
    selfies_vocab = {token: i+1 for i, token in enumerate(all_selfies_tokens)}
    selfies_vocab["[PAD]"] = 0

    print(f"SMILES Vocab Size: {len(smiles_vocab)}")
    print(f"SELFIES Vocab Size: {len(selfies_vocab)}")
    
    return df, smiles_vocab, selfies_vocab

def encode_strings(strings, vocab, max_len, is_selfies=False):
    """Converts strings to a tensor of token IDs with explicit type checking."""
    data = []
    for s in strings:
        if is_selfies:
            tokens = list(sf.split_selfies(s))
        else:
            tokens = list(s)
            
        ids = [vocab.get(t, 0) for t in tokens]
        
        ids += [0] * (max_len - len(ids))
        data.append(ids)
    return torch.tensor(data)

if __name__ == "__main__":
    df, s_vocab, sf_vocab = prepare_data()
    
    # calculate max lengths
    max_s = max(df['smiles'].apply(len))
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))
    
    s_tensors = encode_strings(df['smiles'], s_vocab, max_s, is_selfies=False)
    sf_tensors = encode_strings(df['selfies'], sf_vocab, max_sf, is_selfies=True)
    
    print(f"SMILES Tensor Shape: {s_tensors.shape}") 
    print(f"SELFIES Tensor Shape: {sf_tensors.shape}")