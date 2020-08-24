module Scratchy


module Utils

export clip, decision, accuracy

function clip(x::Real, low=1e-15, hi=1 - 1e-15)
    if x < low
        return low
    elseif x > hi
        return hi
    else
        return x
    end
end

function decision(logits::Array)
	# 0.5 decision threshold
	return Int.(logits .>= 0.5)
end

function accuracy(y::Array, y_hat::Array)
	# Compute accuracy
    return sum(y .== y_hat) / size(y, 1)
end

end  # Utils


module LayerOps

export Operation

struct Operation
    name::String       # The name of this layer operation
    forward::Function  # The function itself
    backward::Function # The derivative of this function
end

end  # LayerOps


module Activations

using ..LayerOps

export Sigmoid, ReLU, Tanh

# Sigmoid (Logistic)
function sigmoid(h::Array{Float64})
    return @. 1.0 / (1.0 + exp(-h))
end

function d_sigmoid(h::Array{Float64})
    forward_vals = sigmoid(h)
    return @. forward_vals * (1.0 - forward_vals)
end

Sigmoid = Operation("Sigmoid", sigmoid, d_sigmoid)


# ReLU
function relu(x)
    return replace(a -> a<0 ? 0 : a, x)
end

function d_relu(x)
    return replace(a -> a<0 ? 0 : 1, x)
end

ReLU = Operation("ReLU", relu, d_relu)


# Tanh
function tanh(x)
    ex = exp.(x)
    enx = exp.(-x)
    return @. (ex - enx) / (ex + enx)
end

function d_tanh(x)
    return @. 1 - (tanh(x)^2)
end

Tanh = Operation("Tanh", tanh, d_tanh)


end  # Activations


module Losses

using ..LayerOps
import ..Utils

export BinaryCrossEntropy

# Binary cross entropy
function binary_crossentropy(y::Array, o::Array{Float64})
    # y: the gold-standard labels 0 or 1
    # o: the real-valued output [0,1]
    n = size(o, 1)
    y_hat = Utils.clip.(o)
    ces = @. y * log(o) + (1 - y) * log(1 - o)
    return -(1 / n) * sum(ces)
end

function d_binary_crossentropy(y::Array, o::Array{Float64})
    # y: the gold-standard labels 0 or 1
    # o: the real-valued output [0,1]
    y_hat = Utils.clip.(o)
    return @. (o - y) / (o * (1 - o))
end

BinaryCrossEntropy = Operation("BinaryCrossEntropy", binary_crossentropy,
                               d_binary_crossentropy)

end  # Losses


module Layers

using ..LayerOps
import Random

export Dense, forward!

# Dense feed-forward layer
mutable struct Dense
    name::String            # What we'll call this layer.
    W::Array{Float64}       # The weight matrix
    b::Array{Float64}       # The bias term
    σ::Operation            # The activation function
    δ::Array{Float64}       # Used in backprop
    input::Array{Float64}   # The inputs from the previous layer
    z::Array{Float64}       # Weighted inputs XW + b
    output::Array{Float64}  # The output of this layer, i.e. the actvations

    # Constructor v1: When we have already initialized W and b
    function Dense(name::String, W::Array{Float64},
				   b::Array{Float64}, σ::Operation)
        if !isa(σ, Operation)
            error("Activation function '$σ' in layer '$name' is not an Operation")
        end
        if size(W, 2) != size(b, 2)
            error("Incompatible W and b shapes in layer '$name'")
        else
            # Since δ, input, and output are computed later,
			# we init empty arrays.
            new(name, W, b, σ, [], [], [], [])
        end
    end
    
    # Constructor v2: When we want to randomly initialize W and
	# b given the shape of W.
    function Dense(name::String, size::Tuple{Int,Int}, σ::Operation)
        W = Random.rand(Float64, size)
        b = Random.rand(Float64, (1, size[2]))
        # Since δ, input, and output are computed later, we init as empty arrays.
        new(name, W, b, σ, [], [], [], [])
    end
end

# Dense layer forward op
function forward!(l::Dense, X::Array)
    # It's always a good idea to double check that the shapes are compatible.
    if size(X, 2) != size(l.W, 1)
        error("Input shape $(size(X)) incompatible with layer '$(l.name)' shape $(size(l.W))")
    end
    l.input = X
    l.z = (X * l.W) .+ l.b  # Weighted inputs
    l.output = l.σ.forward(l.z)
    return l.output
end

end  # Layers


module Models

using ..LayerOps
import ..Layers

export Network, forward!, backward!, fit!, display

struct Network
    name::String
    layers::Array
    loss::Operation
    η::Real  # The learning rate
    
    # Constructor: Check that all layer shapes are compatible
    function Network(name::String, layers::Array, loss::Operation, η::Real)
        l1 = layers[1]
        for l2 in layers[2:end]
            l1_outdim = size(l1.W, 2)
            l2_indim = size(l2.W, 1)
            if l1_outdim != l2_indim
                error("""Output dimension $(size(l1.W, 2)) of layer
                      '$(l1.name)' incompatible with input dimension
                      $(size(l2.W, 1)) of layer '$(l2.name)'.""")
            end
        end
        new(name, layers, loss, η)
    end
end

# Network forward
function forward!(network::Network, X::Array)
    layer_input = X
    for l in network.layers
        layer_input = Layers.forward!(l, layer_input)
    end
    return layer_input
end

# Helper function for backward!
function compute_δ(network::Network, y::Array, i::Int)
    l = network.layers[i]
    dσ_dz = l.σ.backward(l.z)
    # The base case of the iteration: the last layer in the network.
    if i == length(network.layers)
        o = network.layers[i].output
        δ = network.loss.backward(y, o) .* dσ_dz
        return δ
    end
    l_next = network.layers[i+1]
    # l_next.δ should have already been computed because we're iterating
    # backwards through the network's layers in backward!() below.
    δ = (l_next.δ * transpose(l_next.W)) .* dσ_dz
    return δ
end

# Network backprop
function backward!(net::Network, y::Array)
    num_layers = length(net.layers)
    for i in num_layers:-1:1  # Iterate backwards through the network
        l = net.layers[i]
        l.δ = compute_δ(net, y, i)
        ∂C_∂W = transpose(l.δ) * l.input
        l.W = l.W .- (net.η * transpose(∂C_∂W))
    end
end

# Model estimation
function fit!(net::Network, X::Array, y::Array, epochs::Int)
    for i=1:epochs
        o = forward!(net, X)
        backward!(net, y)
        if i == 1 || i % 100 == 0
            loss = net.loss.forward(y, o)
            println("Loss ($i): $(loss)")
        end
    end
end

# Summarize a network's structure.
import Base.display
function display(nn::Network)
    println("Network: $(nn.name)")
    for layer in nn.layers
        println("$(layer.name) : $(size(layer.W)) : $(layer.σ.name)")
    end
    println("Loss: $(nn.loss.name)")
end


end  # Models


end  # Scratchy
