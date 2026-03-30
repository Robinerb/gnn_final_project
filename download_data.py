import pandas as pd
import requests
import io

URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

def get_larger_dataset(n_samples=5000):
    print(f"Download {n_samples} molecules")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        
        # load the full data
        df = pd.read_csv(io.StringIO(response.text))
        
        if 'smiles' not in df.columns:
            df = df.rename(columns={'structure': 'smiles'})
            
        # take a subset for project
        df_subset = df.sample(n=n_samples, random_state=42)
        
        # save
        df_subset[['smiles', 'qed']].to_csv('data/zinc_subset.csv', index=False)
        print(f"saved {len(df_subset)} molecules to data/zinc_subset.csv")
        
    except Exception as e:
        print(f"Download failed with error: {e}")

if __name__ == "__main__":
    get_larger_dataset(5000)