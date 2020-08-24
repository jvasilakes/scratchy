using Pkg
Pkg.activate("../Scratchy")
using Scratchy, Scratchy.Layers, Scratchy.Losses
using Scratchy.Activations, Scratchy.Models


X = [0 0; 1 1; 0 1; 1 0]
y = [0, 0, 1, 1]

layer1 = Dense("layer1", (2, 3), Sigmoid)
layer2 = Dense("layer2", (3, 1), Sigmoid)
net = Network("FFNN", [layer1, layer2], BinaryCrossEntropy, 0.2)
Scratchy.Models.display(net)


fit!(net, X, y, 1000)

logits = Scratchy.Models.forward!(net, X)
preds = Scratchy.Utils.decision(logits)
acc = Scratchy.Utils.accuracy(y, preds)

println("Logits: $(logits)")
println("Predictions: $(preds)")
println("Gold Labels: $(y)")
println("Accuracy: $(acc)")
