import numpy as np
import onnxruntime as ort
import tokenization
import os
from run_onnx_squad import *
import json
import time
import numpy as np
import onnxruntime as ort
from collections import Counter
import string
import re

predict_file = 'dev-v1.1.json'
model = 'fused-bertsquad.onnx'
sess_options = ort.SessionOptions()

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

my_list = []

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input 
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, 
                                                                              max_seq_length, doc_stride, max_query_length)

N = 400  # Number of samples to process

input_ids = input_ids[:N]
input_mask = input_mask[:N]
segment_ids = segment_ids[:N] 

sess_options.enable_profiling = True
session = ort.InferenceSession(model, sess_options)


n = len(input_ids) 
all_results = []
latency = []  
total_tokens = 0  # Initialize total tokens

print("Running inference...")

for idx in range(0, n, batch_size):
    batch_input_ids = input_ids[idx:idx+batch_size]
    batch_input_mask = input_mask[idx:idx+batch_size]
    batch_segment_ids = segment_ids[idx:idx+batch_size]
    batch_unique_ids = np.arange(idx, idx + len(batch_input_ids), dtype=np.int64)

    # Count the total number of tokens in this batch
    total_tokens += sum(len(ids) for ids in batch_input_ids)

    data = {
        "unique_ids_raw_output___9:0": batch_unique_ids,
        "input_ids:0": batch_input_ids,
        "input_mask:0": batch_input_mask,
        "segment_ids:0": batch_segment_ids
    }
    
    start = time.time()
    result = session.run(["unique_ids:0", "unstack:0", "unstack:1"], data)
    batch_latency = time.time() - start
    latency.append(batch_latency)  # Store batch latency

    # profile_file_name = session.end_profiling()

    # print(f"Profile file saved as: {profile_file_name}")
    
    for i in range(len(batch_input_ids)):
        unique_id = batch_unique_ids[i]
        start_logits = [float(x) for x in result[1][i].flat]
        end_logits = [float(x) for x in result[2][i].flat]
        all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

# Calculate average latency per example in milliseconds
average_inference_time_per_example_ms = (sum(latency) * 1000) / N
print("Average OnnxRuntime inference time per example = {} ms".format(format(average_inference_time_per_example_ms, '.2f')))

# Calculate average latency per token in milliseconds
total_latency_ms = sum(latency) * 1000  # Convert total latency to milliseconds
average_latency_per_token_ms = total_latency_ms / total_tokens
print("Average OnnxRuntime inference time per token = {} ms".format(format(average_latency_per_token_ms, '.2f')))

 
 #postprocess
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

write_predictions(eval_examples, extra_data, all_results,
                  n_best_size, max_answer_length,
                  True, output_prediction_file, output_nbest_file)

with open(output_prediction_file, 'r') as file:
    predictions = json.load(file)

#  Filter out empty rows
filtered_predictions = {k: v for k, v in predictions.items() if v != "empty"}

with open(output_prediction_file, 'w') as file:
    json.dump(filtered_predictions, file, indent=4)

with open(output_nbest_file, 'r') as file:
    nbest_predictions = json.load(file)

filtered_nbest_predictions = {k: [ans for ans in v if ans.get('text', '') != "empty"] for k, v in nbest_predictions.items()}

with open(output_nbest_file, 'w') as file:
    json.dump(filtered_nbest_predictions, file, indent=4)

#print results
with open(output_prediction_file) as json_file:  
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))
    
#F1 score
    
def extract_true_answers(squad_data_file):
    true_answers = {}
    with open(squad_data_file, 'r') as f:
        squad_data = json.load(f)
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    question_id = qa['id']
                    true_answers[question_id] = [answer['text'] for answer in qa['answers']]
    return true_answers



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punct(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punct(lower(s))))

def f1_score(prediction, truths):
    """
    Compute the maximum F1 score of the prediction against a list of truth answers.
    """
    # Ensure that truths is a list of strings. If it's not, make it a list.
    if not isinstance(truths, list):
        truths = [truths]

    # Compute F1 scores for the prediction against each truth answer
    f1_scores = [compute_f1(prediction, truth) for truth in truths]

    # Return the maximum F1 score
    return max(f1_scores, default=0)

def compute_f1(prediction, truth):
    """
    Calculate the F1 score between prediction and truth strings.
    """
    prediction_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# Calculate F1 score for each prediction
true_answers = extract_true_answers('dev-v1.1.json')
f1_scores = [f1_score(predictions[qid], true_answers[qid]) for qid in predictions]

# Calculate average F1 score
average_f1 = sum(f1_scores) / len(f1_scores)
print(f"Average F1 score: {average_f1}")

# Calculate average latency per example in milliseconds
average_inference_time_per_example_ms = (sum(latency) * 1000) / N
print("Average OnnxRuntime inference time per example = {} ms".format(format(average_inference_time_per_example_ms, '.2f')))

# Calculate average latency per token in milliseconds
total_latency_ms = sum(latency) * 1000  # Convert total latency to milliseconds
average_latency_per_token_ms = total_latency_ms / total_tokens
print("Average OnnxRuntime inference time per token = {} ms".format(format(average_latency_per_token_ms, '.2f')))

# Assuming the extraction of true answers is done elsewhere and available as `true_answers`

# Calculate F1 scores for filtered predictions
f1_scores = []
for qid in filtered_predictions:
    if qid in true_answers:
        pred_answer = filtered_predictions[qid]
        true_answer_list = true_answers[qid]
        # Calculate the F1 score for this prediction against all true answers for this question
        f1_scores.append(f1_score(pred_answer, true_answer_list))

# Calculate the average F1 score for all valid predictions
average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
print(f"Average F1 score for valid predictions: {average_f1:.4f}")


# After completing all inferences, end profiling to get the profiling file name
profile_file_name = session.end_profiling()

# Print the location of the profiling file or handle it as needed
print(f"Profile file saved as: {profile_file_name}")

