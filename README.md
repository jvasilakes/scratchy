# Scratchy
## From-scratch neural networks in Julia

The motivation for this work was my inability to find a detailed description of gradient
descent / backpropogation that I could clearly translate into code. The result is a guide
to backprop starting from first-principles. This guide is contained in the `notebooks` directory,
which contains a series of Jupyter notebooks with Julia code each step of the way.
The `Scratchy` directory contains a Julia module for building with neural nets that naturally
resulted from the code in the notebooks.

 1. [Backprop for a small linear network][1]
 2. [Backprop for a network with a single hidden layer][2]
 3. [A generalized backprop implementation for feed-forward networks of arbitrary width and depth][3]

Usage:

```julia
using Pkg 
Pkg.activate("../Scratchy")
using Scratchy, Scratchy.Layers, Scratchy.Losses
using Scratchy.Activations, Scratchy.Models


# XOR problem
X = [0 0; 1 1; 0 1; 1 0]
y = [0, 0, 1, 1]

# Create the network
layer1 = Dense("layer1", (2, 3), Sigmoid)
layer2 = Dense("layer2", (3, 1), Sigmoid)
net = Network("FFNN", [layer1, layer2], BinaryCrossEntropy, 0.2)

# Summarize the network structure
Scratchy.Models.display(net)

# Fit the network to the data
fit!(net, X, y, 1000)

# Predict and evaluate
logits = Scratchy.Models.forward!(net, X)
preds = Scratchy.Utils.decision(logits)
acc = Scratchy.Utils.accuracy(y, preds)

println("Logits: $(logits)")
println("Predictions: $(preds)")
println("Gold Labels: $(y)")
println("Accuracy: $(acc)")
```

[1]: https://github.com/jvasilakes/scratchy/notebooks/backprop_example1.ipynb
[2]: https://github.com/jvasilakes/scratchy/notebooks/backprop_example2.ipynb
[3]: https://github.com/jvasilakes/scratchy/notebooks/backprop_example3.ipynb
