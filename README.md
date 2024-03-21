# fused-bert-inference
<ul>
  <li>Dataset file used for the inference is 'dev-v1.1.json' - <a href="https://rajpurkar.github.io/SQuAD-explorer/">SQuAD (Stanford Q&A Dataset)</a></li>
  <li>The vocab file is 'uncased_L-12_H-768_A-12/vocab.txt'</li>
  <li>The tokenization code is 'tokenization.py'</li>
  <li>The inference code is 'test.py' and includes metrics for evaluating the model's performance</li>
  <li>The original BERT onnx model used is <a href="https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bert-squad">'bertsquad-12.onnx'</a> which is a model from the ONNX model zoo trained for Question answering tasks</li>
  <li>The fused model version - 'fused-bertsquad.onnx' - is a version where each MatMul+Add sequence is combined and replaced by a GEMM operator</li>
</ul>
