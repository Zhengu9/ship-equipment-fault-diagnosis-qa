from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, configs):
        self.configs = configs['embedding_model']
        device = self.configs['device']
        model_path = self.configs['model_path']
        self.model = SentenceTransformer(model_path, trust_remote_code=True, local_files_only=True, device=device)
        
    def encode(self, texts, batch_size=64, show_progress_bar=False):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)