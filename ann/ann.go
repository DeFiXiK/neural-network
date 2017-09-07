package ann

import (
	"fmt"
	"math/rand"
)

// Neuron represents a single neuron in a network.
type Neuron struct {
	Weights []float64
	Bias    float64
}

// Execute calculates neuron's output with given input values.
func (n Neuron) Execute(inputs []float64) float64 {
	if len(inputs) != len(n.Weights) {
		panic(fmt.Sprintf(
			"Number of input values (%v) does not match number of weights (%v)",
			len(inputs), len(n.Weights),
		))
	}

	out := 0.0
	for i := range inputs {
		out += inputs[i] * n.Weights[i]
	}
	out += n.Bias

	return out
}

// Network contains layers on neurons.
type Network struct {
	InputCount int
	Layers     [][]Neuron
}

// New constructs a new ANN with given number of inputs and numbers of neuron
// on each layer.
func New(inputs int, counts ...int) *Network {
	network := &Network{InputCount: inputs}
	for _, cnt := range counts {
		network.AddLayer(cnt)
	}
	return network
}

// AddLayer creates a new layer of `count` neurons and appends to the network.
func (n *Network) AddLayer(count int) {
	var prevOut int
	if len(n.Layers) == 0 {
		prevOut = n.InputCount
	} else {
		prevOut = len(n.Layers[len(n.Layers)-1])
	}

	var newLayer []Neuron
	for i := 0; i < count; i++ {
		weights := make([]float64, prevOut)
		for i := range weights {
			weights[i] = rand.Float64()*2 - 1
		}

		var neuron = Neuron{
			Weights: weights,
			Bias:    rand.Float64()*2 - 1,
		}
		newLayer = append(newLayer, neuron)
	}

	n.Layers = append(n.Layers, newLayer)
}

// Execute applies given input values to the networks and returns output values.
func (n *Network) Execute(inputs []float64) []float64 {
	var lastOutput = inputs
	for _, layer := range n.Layers {
		output := make([]float64, len(layer))
		for i := range layer {
			neuronOut := layer[i].Execute(lastOutput)
			output[i] = neuronOut
		}
		lastOutput = output
	}
	return lastOutput
}
