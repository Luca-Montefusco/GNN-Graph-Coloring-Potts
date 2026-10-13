# GNN-Graph-Coloring-Potts
Project completed during the third year of the B.Sc. in "Mathematical Sciences for Artificial Intelligence" for the Physic application to Machine Learning course.

Unsupervised graph coloring with Graph Neural Networks: a continuous relaxation of the antiferromagnetic Potts model, comparing GraphSAGE and GCN architectures (Sapienza University of Rome, 2026)

This project tackles the graph coloring problem (GCP), an NP-hard combinatorial problem, as a differentiable optimization task. Instead of training a global predictive model, a GNN is used as a per-graph variational optimizer: each node gets a trainable embedding, and the network outputs a probability distribution over k colors. The loss combines a continuous relaxation of the antiferromagnetic Potts Hamiltonian, which penalizes adjacent nodes sharing the same color, with an entropy-maximization term that prevents the distributions from collapsing onto a degenerate solution:

$$\mathcal{L} = \lambda_H \sum_{(i,j) \in E} \mathbf{p}_i \cdot \mathbf{p}_j \;-\; \lambda_E \, H(\mathbf{p})$$

After training, colors are extracted with argmax and the solution is evaluated by its number of hard conflicts. Two message-passing architectures are compared under the same loss and learning rate: SAGEGNN (2 × SAGEConv) and PairGNN (3 × GCNConv). The chromatic number is estimated by sweeping k from 2 to 15 over 5 random seeds, and results are compared against the classical DSATUR heuristic on the DIMACS instances g9 (49 nodes, 476 edges) and g15 (188 nodes, 3885 edges).

## Usage
Open the notebook (designed for Google Colab, GPU recommended) and install the dependencies (pip install -r requirements.txt).
Run the first cells to download the DIMACS .col graphs from the course repository.
Run the training cells: both models are trained for every k in [2, 15] and every seed (about 17 minutes on GPU), then the statistics, DSATUR baseline and plots are produced.

## Author
Luca Montefusco, Matteo Pinato, Sofia De Angelis, Aurora Giaccherini
Sapienza University of Rome, 2026
