import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.models import TfIdfEmbedder, CountVectorizerEmbedder, SentenceTransformerSmall, SentenceTransformerLarge

tqdm.pandas()

PARLAMINT_DATASET_SELECTION = [
    "PER_SENTENCE",
    "PER_UTTERANCE"
]

EMBEDDING_MODEL_SELECTION = {
    "SENTENCE_TRANSFORMER_LARGE": SentenceTransformerLarge,
    "SENTENCE_TRANSFORMER_SMALL": SentenceTransformerSmall,
    "TFIDF_EMBEDDER": TfIdfEmbedder,
    "COUNT_VECTORIZER": CountVectorizerEmbedder,
}

# TODO select dataset (based on PARLAMINT_DATASET_SELECTION), model (based on EMBEDDING_MODEL_SELECTION) and output filename
SELECTED_DATASET = "PER_UTTERANCE"
SELECTED_MODEL = "SENTENCE_TRANSFORMER_LARGE"
EMBEDDINGS_OUTPUT_FILEPATH = f"./data/{SELECTED_MODEL}_{SELECTED_DATASET}.pkl"


def load_dataset_parlamint(file_path, selected_dataset):
    # Load parlamint dataset
    df_parlamint = pd.read_csv(file_path, sep="\t")
    # df_parlamint = df_parlamint.head(2000)

    if selected_dataset == "PER_SENTENCE":
        return df_parlamint
    # Group sentence by utterance (=Parent_ID)
    df_parlamint_grouped = (df_parlamint.groupby(["Parent_ID"])["Text"]
                            .apply(lambda s: " ".join(s))
                            .reset_index(name="Text"))
    return df_parlamint_grouped


def main():
    print(f"Loading dataset {SELECTED_DATASET}...")
    df_parlamint = load_dataset_parlamint("../materials/parlamint/parlamint-it-is-2022.txt", SELECTED_DATASET)

    print(f"Initializing model {SELECTED_MODEL}...")
    if SELECTED_MODEL == "SENTENCE_TRANSFORMER_LARGE":
        model = SentenceTransformerLarge()
    elif SELECTED_MODEL == "SENTENCE_TRANSFORMER_SMALL":
        model = SentenceTransformerSmall()
    elif SELECTED_MODEL == "TFIDF_EMBEDDER":
        model = TfIdfEmbedder(vocabulary=df_parlamint["Text"].to_list(), min_df=10,
                              stop_words='english')
    elif SELECTED_MODEL == "COUNT_VECTORIZER":
        model = CountVectorizerEmbedder(
            vocabulary=df_parlamint["Text"].to_list(), min_df=10, stop_words='english',
            n_gram_range=(1, 3))
    else:
        raise Exception(f"Selected model {SELECTED_MODEL} is not supported.")

    print("Generating embeddings...")
    data_embeddings = model.embed(df_parlamint["Text"].to_list())
    df_parlamint["embedding"] = list(
        data_embeddings if type(data_embeddings) is np.ndarray else data_embeddings.toarray())

    print("Write embedding file...")
    dir_name = os.path.dirname(EMBEDDINGS_OUTPUT_FILEPATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df_parlamint.to_pickle(EMBEDDINGS_OUTPUT_FILEPATH)
    print(f"Processing finished for {SELECTED_DATASET} dataset and {SELECTED_MODEL} model...")


if __name__ == "__main__":
    main()
