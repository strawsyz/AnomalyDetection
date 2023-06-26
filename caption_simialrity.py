from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import torch


def sentence_similarity(s1: str, s2: str, model=None):
    '''
    比较两个句子的相似度，交换位置不会有影响
    Args:
        s1: such as, That is a happy person
        s2: such as, That is a happy dog
        model:

    Returns:
        score
    '''
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode([s1, s2,s2], convert_to_tensor=True)
    print(embeddings.shape)
    print(embeddings)
    res = cosine_similarity(torch.unsqueeze(embeddings[0], dim=0), torch.unsqueeze(embeddings[1], dim=0))
    return res


if __name__ == '__main__':
    with torch.no_grad():
        cos = sentence_similarity("He is a happy dog", "He is a sad dog")
        print(cos)
    from sentence_transformers import SentenceTransformer, util
    util.dot_score()