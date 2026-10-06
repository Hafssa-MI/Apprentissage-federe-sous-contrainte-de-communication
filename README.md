# Federated learning under a communication budget

Quantization, partial participation and robust aggregation on non-IID data, simulated on the UCI HAR smartphone activity-recognition dataset (course project, subject EMB-5).

**Research question.** For a fixed communication budget, which combination of update quantization, partial participation and robust aggregation best preserves convergence on non-IID data, including when some clients are corrupted?

Everything is in one notebook: [`federated_learning_har.ipynb`](federated_learning_har.ipynb). It runs a simulation of one server and 21 clients (PyTorch, CPU) and compares configurations at the same number of uploaded bytes, with 3 seeds.

## Main findings
(Setup: HAR, a small MLP, 21 clients, 3 seeds; see the notebook for the numbers, caveats and limits.)

- Plain FedAvg reaches 0.925 to 0.941 accuracy on unseen subjects against 0.945 for centralized training. Data heterogeneity mainly slows convergence (81 MB to reach 0.90 at alpha = 10, 230 MB at alpha = 0.1).
- At a fixed byte budget, partial participation and quantization are large savings: with 20% of the clients and 4-bit messages a round costs about 40 times fewer bytes, and 10 MB reach 0.925 in the natural split against 0.842 with 32-bit messages. The gain shrinks as the budget grows.
- Corrupted clients (10%) break plain FedAvg under an amplified sign flip (accuracy 0.168 at x10 in the natural split); random noise and label flipping hurt much less.
- Robust aggregation helps at a price that depends on heterogeneity: with similar clients the trimmed mean and median recover about 0.89 to 0.91; with strong label skew the median fails even without attackers and the trimmed mean works only if it trims at least as many values as there are corrupted clients.
- There is no single best configuration; the best choice depends on whether attackers are present and on how heterogeneous the data is. The direction of the findings is stable across model size, learning rate and local epochs.

## Run it
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab federated_learning_har.ipynb
```
The first cells download the HAR dataset (internet needed). On a CPU, the larger grids take roughly 50 minutes (Step 8), 20 minutes (Step 9), 90 minutes (Step 10) and 45 minutes (Step 12). Finished runs of these steps are cached in `.pkl` files and skipped on rerun.

## Limits
One dataset and one small model, 3 seeds, 10% corrupted clients with non-adaptive attacks, only uploads counted in the byte budget, FedProx not implemented. The notebook lists all limitations and future work.

## Author
_add your name_
