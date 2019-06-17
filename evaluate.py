import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def eval_emb_metrics(hypothesis, references, embedding_array):
    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []
    for hyp in hypothesis:
        embs = np.array([embedding_array[word] for word in hyp])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb)))

        emb_hyps.append(embs)
        avg_emb_hyps.append(avg_emb)
        extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for ref in references:
        embs = np.array([embedding_array[word] for word in ref])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        #avg_emb = np.mean(embs,axis=0)
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb)))
        emb_refs.append(embs)
        avg_emb_refs.append(avg_emb)
        extreme_emb_refs.append(extreme_emb)

    avg_cos_similarity = np.array([cos_sim(hyp,ref) for hyp,ref in zip(avg_emb_hyps,avg_emb_refs)])
    avg_cos_similarity = avg_cos_similarity.mean()
    extreme_cos_similarity = np.array([cos_sim(hyp, ref) for hyp, ref in zip(extreme_emb_hyps, extreme_emb_refs)])
    extreme_cos_similarity = extreme_cos_similarity.mean()

    scores = []
    for emb_ref, emb_hyp in zip(emb_refs, emb_hyps):
        simi_matrix = cosine_similarity(emb_ref, emb_hyp)
        dir1 = simi_matrix.max(axis=0).mean()
        dir2 = simi_matrix.max(axis=1).mean()
        scores.append((dir1+dir2)/2)
    greedy_scores = np.mean(scores)

    return avg_cos_similarity,extreme_cos_similarity,greedy_scores

def diversity_metrics(hypothesis):
    unigram_list = list(itertools.chain(*hypothesis))
    total_num_unigram = len(unigram_list)
    unique_num_unigram = len(set(unigram_list))
    bigram_list = []
    for hyp in hypothesis:
        hyp_bigram_list = list(zip(hyp[:-1],hyp[1:]))
        bigram_list += hyp_bigram_list
    total_num_bigram = len(bigram_list)
    unique_num_bigram = len(set(bigram_list))
    dist_1 = unique_num_unigram/total_num_unigram
    dist_2 = unique_num_bigram/total_num_bigram
    unfrequent_num_unigram = np.sum(np.array(unigram_list) > 2000)
    novelty = unfrequent_num_unigram/total_num_unigram
    return dist_1,dist_2,novelty

def evaluate_response(hypothesis,references,embedding_array):
    if not embedding_array is None:
        avg_cos_similarity, extreme_cos_similarity, greedy_scores = eval_emb_metrics(hypothesis, references, embedding_array)
    h_dist_1,h_dist_2,h_novelty = diversity_metrics(hypothesis)
    r_dist_1,r_dist_2,r_novelty = diversity_metrics(references)
    if not embedding_array is None:
        generation_dict = {"emb_avg": avg_cos_similarity, "emb_ext": extreme_cos_similarity,
                           "emb_gre": greedy_scores, "dist-1": h_dist_1, "dist-2": h_dist_2, "novel": h_novelty}
        print("EmbeddingAverageCosineSimilairty: {0:.6f}".format(avg_cos_similarity))
        print("EmbeddingExtremeCosineSimilairty: {0:.6f}".format(extreme_cos_similarity))
        print("GreedyMatchingScore: {0:.6f}".format(greedy_scores))
    else:
        generation_dict = {"dist-1": h_dist_1, "dist-2": h_dist_2, "novel": h_novelty}
        
    print("Dist-1 : Preeicted {0:.6f} True {1:.6f}".format(h_dist_1,r_dist_1))
    print("Dist-2 : Predicted {0:.6f} True {1:.6f}".format(h_dist_2,r_dist_2))
    print("Novelty: Predicted {0:.6f} True {1:.6f}".format(h_novelty,r_novelty))
    print("\n")
    return generation_dict



