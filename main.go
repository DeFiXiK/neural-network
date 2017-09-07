package main

import (
	"fmt"

	"github.com/DeFiXiK/neural-network/ann"
)

func main() {
	network := ann.New(4, 3, 2)
	fmt.Printf("%#v\n", network.Execute([]float64{1, -1, 1, -1}))
}
