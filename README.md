# fused-bert-inference
<ul>
  <li>dev-v1.1.json - SQuAD (Stanford Q&A Dataset)</li>
  <li>uncased_L-12_H-768_A-12/vocab.txt - vocab file</li>
  <li>bertsquad-12.onnx - original BERT onnx model (https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx)</li>
  <li>fused-bertsquad.onnx - fused model where every MatMul+Add sequence is replaced by a GEMM operator</li>
</ul>
