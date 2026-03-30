import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from model_qed import MolecularVAE_QED
from preprocess import prepare_data, encode_strings

def generate_noise_jump(idx=50, steps=5):
    df, s_vocab, sf_vocab = prepare_data()
    max_sf = max(df['selfies'].apply(lambda x: len(list(sf.split_selfies(x)))))
    model = MolecularVAE_QED(len(sf_vocab), max_sf)
    model.load_state_dict(torch.load("checkpoints/selfies_vae_qed.pth"))
    model.eval()

    # starting molecule
    initial_sf = df['selfies'][idx]
    tensor = encode_strings([initial_sf], sf_vocab, max_sf, is_selfies=True)
    
    with torch.no_grad():
        mu, _ = model.encode(tensor)
        z_start = mu.clone().detach()

    walk_smiles = []
    inv_vocab = {v: k for k, v in sf_vocab.items()}
    
    print(f"Starting Jump from: {initial_sf}")
    for i in range(steps):
        # increasing amounts of noise to force structural change
        noise = torch.randn_like(z_start) * (i * 1.5) 
        z_step = z_start + noise
        
        with torch.no_grad():
            logits = model.decode(z_step.view(1, -1))
            token_ids = torch.argmax(logits, dim=-1).squeeze(0)
            tokens = [inv_vocab[idx.item()] for idx in token_ids if idx.item() != 0]
            gen_sf = "".join(tokens)
            sm = sf.decoder(gen_sf)
            walk_smiles.append(sm)

    # save the grid
    mols = [Chem.MolFromSmiles(s) for s in walk_smiles]
    valid_mols = [m for m in mols if m is not None]
    if valid_mols:
        img = Draw.MolsToGridImage(valid_mols, molsPerRow=len(valid_mols), subImgSize=(300, 300))
        img.save('figures/diverse_jump_walk.png')
        print("Success")
    else:
        print("No valid molecules found to draw.")

if __name__ == "__main__":
    generate_noise_jump()