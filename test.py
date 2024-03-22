import os
import numpy as np
import onnxruntime as ort
import tokenization
from run_onnx_squad import *
import json
import time
from collections import Counter
import string
import re

predict_file = 'dev-v1.1.json' # QA Dataset for inference
model = 'fused-bertsquad.onnx' # BERT model
# Use GPU if available, if not default to CPU
sess_options = ort.SessionOptions()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Use read_squad_examples method from run_onnx_squad to read the samples from the dataset
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256 # Max lenght of input sequence (question+context)
doc_stride = 128 # Size of sliding window when the context is longer than max_seq_lenght
max_query_length = 64 # Max lenght of the question
batch_size = 400 # How many samples are processed together in one forward pass
n_best_size = 20 # After predicting the possible answers, how many of the top-ranked answers to keep
max_answer_length = 30 # Max lenght of the answer the model can  generate

# Vocabulary file used by the tokenizer
vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
# Initialize tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

# Use convert_examples_to_features method from run_onnx_squad to convert raw examples into the format expected by the model (input IDs, masks, and segment IDs)
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer, 
                                                                              max_seq_length, doc_stride, max_query_length)

samples_to_process = 400  # Number of total samples from the dataset to process

input_ids = input_ids[:samples_to_process] # input_ids holds token IDs for each input sequence
input_mask = input_mask[:samples_to_process] # input_mask is used to differentiate real tokens from padding tokens
segment_ids = segment_ids[:samples_to_process] # segment_ids is used to distinguish between the question and context sequences

sess_options.enable_profiling = True # Enable profiling to analyze performance
session = ort.InferenceSession(model, sess_options, providers=providers) # Initialize inference session

all_results = [] # Initialize empty list to store the results from the inference for each batch

print("Active execution providers:", session.get_providers()) # Print whether ONNX session is using GPU or CPU
print("Running inference...")

# Start the timer for total inference time measurement
start_time_total = time.time()

# Start a loop to process all batches
for idx in range(0, samples_to_process, batch_size):
    # Slice the input_ids for the current batch
    batch_input_ids = input_ids[idx:idx+batch_size]
    # Slice the input_mask for the current batch
    batch_input_mask = input_mask[idx:idx+batch_size]
    # Slice the segment_ids for the current batch
    batch_segment_ids = segment_ids[idx:idx+batch_size]
    # Generate unique IDs for each sample in the batch
    batch_unique_ids = np.arange(idx, idx + len(batch_input_ids), dtype=np.int64)

    # Prepare the data for the model inference
    data = {
        "unique_ids_raw_output___9:0": batch_unique_ids,  # Array of unique IDs for tracking each input sample
        "input_ids:0": batch_input_ids,                   # Token IDs for each sequence in the batch
        "input_mask:0": batch_input_mask,                 # Mask to differentiate real tokens from padding
        "segment_ids:0": batch_segment_ids                # IDs to distinguish between different segments (question vs. context)
    }

    # Run inference for current batch and collect outputs
    result = session.run(["unique_ids:0", "unstack:0", "unstack:1"], data)

    for i in range(len(batch_input_ids)):
        # Retrieve the unique ID for the current input sample
        unique_id = batch_unique_ids[i]
        # Convert the start logits from the model output to a list of floats
        start_logits = [float(x) for x in result[1][i].flat]
        # Convert the end logits from the model output to a list of floats
        end_logits = [float(x) for x in result[2][i].flat]
        # Append the results for this input sample to the all_results list
        all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

# Mark the end of the inference process, after completing all batches
end_time_total = time.time()

# Define the directory where prediction files will be saved
output_dir = 'predictions'
# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)
# Define the path for the file that will store the final predictions
output_prediction_file = os.path.join(output_dir, "predictions.json")
# Define the path for the file that will store the top N best predictions
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

# Call the function to write predictions and their details into the specified files
write_predictions(eval_examples, extra_data, all_results,
                  n_best_size, max_answer_length,
                  True, output_prediction_file, output_nbest_file)

# After predictions are written, open the predictions file to load the predictions into a variable
with open(output_prediction_file, 'r') as file:
    predictions = json.load(file)

# Filter out predictions that have been marked as "empty"
filtered_predictions = {k: v for k, v in predictions.items() if v != "empty"}

# Overwrite the predictions file with the filtered predictions
with open(output_prediction_file, 'w') as file:
    json.dump(filtered_predictions, file, indent=4)

# Load the n-best predictions from file
with open(output_nbest_file, 'r') as file:
    nbest_predictions = json.load(file)

# Filter out any "empty" answers from the n-best predictions for each question
filtered_nbest_predictions = {
    k: [ans for ans in v if ans.get('text', '') != "empty"]
    for k, v in nbest_predictions.items()
}

# Overwrite the n-best predictions file with the filtered n-best predictions
with open(output_nbest_file, 'w') as file:
    json.dump(filtered_nbest_predictions, file, indent=4)

# Print the filtered predictions for review
with open(output_prediction_file) as json_file:  
    test_data = json.load(json_file)
    print(json.dumps(test_data, indent=2))

# Compute F1 score    
# Function to extract the true answers from a SQuAD dataset file
def extract_true_answers(squad_data_file):
    """
    Extracts the true answers for each question in the SQuAD dataset.
    
    Parameters:
    - squad_data_file: str. The file path to the SQuAD dataset JSON file.
    
    Returns:
    - true_answers: dict. A dictionary where each key is a question ID and the value is a list of true answers.
    """
    true_answers = {}
    # Open and load the SQuAD dataset file
    with open(squad_data_file, 'r') as f:
        squad_data = json.load(f)
        # Iterate through each article in the dataset
        for article in squad_data['data']:
            # Iterate through each paragraph in the article
            for paragraph in article['paragraphs']:
                # Iterate through each question-answer set in the paragraph
                for qa in paragraph['qas']:
                    question_id = qa['id']  # Extract the question ID
                    # Extract all the answers for this question
                    true_answers[question_id] = [answer['text'] for answer in qa['answers']]
    return true_answers

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace from a string."""
    def remove_articles(text):
        # Remove English articles ('a', 'an', 'the') from the text
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # Remove extra white spaces from the text
        return ' '.join(text.split())

    def remove_punct(text):
        # Remove all punctuation from the text
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        # Convert all characters in the text to lowercase
        return text.lower()

    # Apply all normalization functions to the input string in order
    return white_space_fix(remove_articles(remove_punct(lower(s))))

def f1_score(prediction, truths):
    """
    Compute the maximum F1 score of a prediction against a list of truth answers.
    
    Parameters:
    - prediction: str. The predicted answer.
    - truths: list of str. The list of true answers.
    
    Returns:
    - The highest F1 score achieved between the prediction and any of the true answers.
    """
    # Ensure that truths is a list to facilitate iteration
    if not isinstance(truths, list):
        truths = [truths]

    # Compute F1 scores for the prediction against each truth answer
    f1_scores = [compute_f1(prediction, truth) for truth in truths]

    # Return the highest F1 score among all comparisons
    return max(f1_scores, default=0)

def compute_f1(prediction, truth):
    """
    Calculate the F1 score between a single prediction and a truth string.
    
    Parameters:
    - prediction: str. The predicted answer.
    - truth: str. The true answer.
    
    Returns:
    - f1: float. The F1 score between prediction and truth.
    """
    # Normalize both prediction and truth for fair comparison
    prediction_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()

    # Find the common tokens between prediction and truth
    common = Counter(prediction_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    # If there are no common tokens, return F1 score of 0
    if num_same == 0:
        return 0

    # Calculate precision and recall
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(truth_tokens)

    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Extract true answers for comparison
true_answers = extract_true_answers(predict_file)
# Calculate F1 scores for each prediction, comparing with the true answers
f1_scores = [f1_score(predictions[qid], true_answers[qid]) for qid in predictions]

# Initialize a list to store F1 scores for filtered predictions
f1_scores = []
# Iterate through each question ID in the filtered predictions
for qid in filtered_predictions:
    # Check if the question ID exists in the true answers
    if qid in true_answers:
        # Retrieve the predicted answer for the current question ID
        pred_answer = filtered_predictions[qid]
        # Retrieve the list of true answers for the current question ID
        true_answer_list = true_answers[qid]
        # Calculate and append the F1 score for this prediction against the true answers
        f1_scores.append(f1_score(pred_answer, true_answer_list))

# Calculate the average F1 score for all valid predictions; if there are no scores, default to 0
average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
# Display the calculated average F1 score with 4 decimal places
print(f"Average F1 score for valid predictions: {average_f1:.4f}")

# After completing all inference tasks, end the profiling session to save profiling data
profile_file_name = session.end_profiling()
# Inform about the location or name of the saved profiling file
print(f"Profile file saved as: {profile_file_name}")

# Calculate the total inference time in microseconds by subtracting the start time from the end time and converting to microseconds
total_inference_time_microseconds = (end_time_total - start_time_total) * 1e6
# Display the total inference time for all examples
print(f"Total inference time for all examples: {total_inference_time_microseconds} μs")

# Calculate the average inference time per example by dividing the total inference time by the number of examples (N)
average_inference_time_per_example_microseconds = total_inference_time_microseconds / samples_to_process
# Display the average inference time per example in microseconds with two decimal places
print(f"Average inference time per example: {average_inference_time_per_example_microseconds:.2f} μs")

