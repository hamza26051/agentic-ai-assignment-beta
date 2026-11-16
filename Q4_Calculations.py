from math import sqrt, log, exp
from collections import Counter
from statistics import mean, stdev

stage1 = [[100, 5, 15], [8, 80, 12], [20, 10, 50]]
stage2 = [[106, 4, 10], [6, 84, 10], [9, 7, 64]]
labels = ["Positive", "Negative", "Neutral"]

def wilson(p, n, z=1.96):
    d = 1 + z*z/n
    m = (p + z*z/(2*n))/d
    h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return m-h, m+h

def evaluate(matrix):
    n = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(3))
    per_class = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        per_class.append({"class": label, "TP": tp, "FP": fp, "FN": fn,
                          "precision": tp/(tp+fp), "recall": tp/(tp+fn),
                          "F1": 2*tp/(2*tp+fp+fn), "support": sum(matrix[i])})
    accuracy = correct/n
    return {"accuracy": accuracy, "95% Wilson CI": wilson(accuracy, n),
            "per_class": per_class, "macro_F1": mean(x["F1"] for x in per_class),
            "weighted_F1": sum(x["F1"]*x["support"] for x in per_class)/n,
            "micro_F1": accuracy}

first = evaluate(stage1)
second = evaluate(stage2)
first, second

orders = [0.847, 0.840, 0.853, 0.837, 0.850]
{"mean": mean(orders), "sample_standard_deviation": stdev(orders),
 "range": max(orders)-min(orders),
 "accuracy_gain_in_percentage_points": 100*(second["accuracy"]-first["accuracy"])}

reference = "the delivery was late but the product quality is good".split()
candidate = "the product quality is good but delivery was slow".split()

def ngrams(words, n):
    return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))

def overlap(n):
    ref = ngrams(reference, n)
    pred = ngrams(candidate, n)
    return sum(min(count, ref[k]) for k, count in pred.items())

def rouge_n(n):
    matches = overlap(n)
    precision = matches/(len(candidate)-n+1)
    recall = matches/(len(reference)-n+1)
    return {"precision":precision,"recall":recall,"F1":2*precision*recall/(precision+recall)}

bleu_precisions = [overlap(n)/(len(candidate)-n+1) for n in range(1,5)]
brevity_penalty = min(1, exp(1-len(reference)/len(candidate)))
bleu4 = brevity_penalty*exp(sum(log(p) for p in bleu_precisions)/4)

dp = [[0]*(len(candidate)+1) for _ in range(len(reference)+1)]
for i in range(1, len(reference)+1):
    for j in range(1, len(candidate)+1):
        dp[i][j] = dp[i-1][j-1]+1 if reference[i-1] == candidate[j-1] else max(dp[i-1][j], dp[i][j-1])
lcs_length = dp[-1][-1]
lcs_precision = lcs_length/len(candidate)
lcs_recall = lcs_length/len(reference)
rouge_l = 2*lcs_precision*lcs_recall/(lcs_precision+lcs_recall)

remaining = list(enumerate(reference))
matches = []
for i, word in list(enumerate(candidate))[::-1]:
    for j, ref_word in remaining[::-1]:
        if word == ref_word:
            matches.append((i,j))
            remaining.remove((j,ref_word))
            break
matches.sort()
chunks = 1 + sum(not (matches[i][0] == matches[i-1][0]+1 and matches[i][1] == matches[i-1][1]+1) for i in range(1,len(matches)))
meteor_precision = len(matches)/len(candidate)
meteor_recall = len(matches)/len(reference)
meteor_fmean = meteor_precision*meteor_recall/(0.9*meteor_precision+0.1*meteor_recall)
meteor_penalty = 0.5*(chunks/len(matches))**3
meteor = (1-meteor_penalty)*meteor_fmean
{"BLEU4": bleu4, "BLEU_modified_precisions": bleu_precisions, "brevity_penalty": brevity_penalty,
 "ROUGE1": rouge_n(1), "ROUGE2": rouge_n(2), "ROUGE_L_F1": rouge_l,
 "METEOR_exact_stem": meteor, "METEOR_matching_tokens": len(matches), "METEOR_chunks": chunks}

token_probabilities = [0.60, 0.40, 0.50, 0.80, 0.70, 0.30, 0.25, 0.50, 0.20]
negative_log_sum = -sum(log(p) for p in token_probabilities)
mean_negative_log = negative_log_sum/len(token_probabilities)
perplexity = exp(mean_negative_log)
{"sum_negative_log": negative_log_sum, "mean_negative_log": mean_negative_log, "perplexity": perplexity}
