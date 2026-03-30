import torch
import pandas as pd
import selfies as sf
from rdkit import Chem
from model import MolecularVAE
from preprocess import prepare_data

def is_valid(string, is_selfies=False):
    """Checks if a molecule string is chemically valid."""
    if is_selfies:
        try:
            # SELFIES to SMILES, then check with RDKit
            smiles = sf.decoder(string)
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    else:
        mol = Chem.MolFromSmiles(string)
        return mol is not None

def generate_and_validate(model, vocab, max_len, is_selfies=False, n_samples=10):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    valid_count = 0
    
    with torch.no_grad():
        # sample random vectors from the Latent Space
        z = torch.randn(n_samples, 64)
        
        # pass through the decoder
        z_hidden = model.latent_to_hidden(z).unsqueeze(0)
        
        decoder_input = torch.zeros(n_samples, max_len, 128)
        logits, _ = model.decoder_gru(decoder_input, z_hidden)
        logits = model.fc_out(logits)
        
        # convert logits to actual strings
        token_ids = torch.argmax(logits, dim=-1)
        
        for i in range(n_samples):
            ids = token_ids[i].tolist()
            tokens = [inv_vocab[idx] for idx in ids if idx != 0]
            generated_str = "".join(tokens)
            
            # check Validity
            if is_valid(generated_str, is_selfies):
                valid_count += 1
                status = "VALID"
            else:
                status = "INVALID"
            
            print(f"Sample {i+1}: {generated_str[:40]}... [{status}]")
            
    print(f"Total Validity: {valid_count}/{n_samples}")

if __name__ == "__main__":
    df, s_vocab, sf_vocab = prepare_data()
    
    # calculate max lengths again
    max_s = max(df['smiles'].apply(len))
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))

    # load and test SMILES
    sm_model = MolecularVAE(len(s_vocab), max_s)
    sm_model.load_state_dict(torch.load("checkpoints/smiles_vae.pth"))
    generate_and_validate(sm_model, s_vocab, max_s, is_selfies=False)

    # load and test SELFIES
    sf_model = MolecularVAE(len(sf_vocab), max_sf)
    sf_model.load_state_dict(torch.load("checkpoints/selfies_vae.pth"))
    generate_and_validate(sf_model, sf_vocab, max_sf, is_selfies=True)