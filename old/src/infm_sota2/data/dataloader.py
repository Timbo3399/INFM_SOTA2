from torch.utils.data import DataLoader

from infm_sota2.data.dataset import BPEDataset
from infm_sota2.data.embedding import Embeddings


def create_dataloader(
    text: str,
    seq_len: int = 128,
    batch_size: int = 32,
    stride: int | None = None,
    emb_dim: int = 256,
    shuffle: bool = True,
):
    """Erstellt Dataset, DataLoader und Embeddings in einem Schritt."""

    dataset = BPEDataset(text, seq_len=seq_len, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    embeddings = Embeddings(dataset.vocab_size, emb_dim=emb_dim, max_len=seq_len)

    return dataloader, embeddings, dataset
