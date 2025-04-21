def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0.0  # Avoid division by zero
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    
    if tp + fn == 0:
        return 0.0  # Avoid division by zero
    return tp / (tp + fn)

def calculate_f1_score(precision, recall):

    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

def evaluate_retrieval(relevant_docs, retrieved_docs):
    
    tp = len(relevant_docs & retrieved_docs)  # True Positives (Correctly retrieved relevant docs)
    fp = len(retrieved_docs - relevant_docs)  # False Positives (Retrieved but not relevant)
    fn = len(relevant_docs - retrieved_docs)  # False Negatives (Relevant but not retrieved)

    # Calculate metrics
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1_score = calculate_f1_score(precision, recall)

    return tp, fp, fn, precision, recall, f1_score

# Example Data: Ground Truth & Retrieved Documents
relevant_documents = {1, 2, 3, 5, 7}  # Actual relevant document IDs
retrieved_documents = {1, 2, 4, 5, 6}  # Documents retrieved by the IR system

# Compute evaluation metrics
tp, fp, fn, precision, recall, f1_score = evaluate_retrieval(relevant_documents, retrieved_documents)

# Display Results
print("\n=== Evaluation Metrics for Information Retrieval ===")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")