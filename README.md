# fused-bert-inference
<ul>
  <li>Dataset file used for the inference is 'dev-v1.1.json' - SQuAD (Stanford Q&A Dataset)</li>
  <li>The vocab file is 'uncased_L-12_H-768_A-12/vocab.txt'</li>
  <li>The inference code is in 'test.py'</li>
  <li>The onnx models included are the original BERT onnx model - 'bertsquad-12.onnx' (https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx) and a fused version - 'fused-bertsquad.onnx' - where every MatMul+Add sequence is replaced by a GEMM operator</li>
</ul>
