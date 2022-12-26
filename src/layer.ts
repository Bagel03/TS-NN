import { Activation } from "./activation.js";
import { Cost } from "./cost.js";
import { DataPointRunData } from "./data.js";
import { arrFromFunc, randomInNormalDistribution } from "./utils.js";

export class Layer {
    weights: number[];
    biases: number[];

    weightGradients: number[];
    biasGradients: number[];

    weightVelocities: number[];
    biasVelocities: number[];

    constructor(
        public nodes: number,
        public inputs: number,
        public activation: Activation,
        public idx: number,
    ) {
        this.biases = arrFromFunc(nodes, () => Math.random() - 0.5);

        const sqrt = Math.sqrt(inputs);
        this.weights = arrFromFunc(nodes * inputs, () => {
            let y = randomInNormalDistribution(1, 0);
            return y / sqrt;
        });

        this.weightGradients = new Array(nodes * inputs).fill(0);
        this.biasGradients = new Array(nodes).fill(0);
        this.biasVelocities = new Array(nodes).fill(0);
        this.weightVelocities = new Array(nodes * inputs).fill(0);
    }

    private getWeightIdx(node: number, input: number) {
        return node * this.inputs + input;
    }

    private getWeight(node: number, input: number) {
        return this.weights[this.getWeightIdx(node, input)];
    }

    feedForward(data: DataPointRunData) {
        let inputs = this.idx == 0 ? data.inputs : data.activations[this.idx - 1];

        for (let node = 0; node < this.nodes; node++) {
            let sum = this.biases[node];

            for (let input = 0; input < this.inputs; input++) {
                sum += inputs[input] * this.getWeight(node, input);
            }

            data.weightedSums[this.idx][node] = sum;
        }

        this.activation.function(
            data.weightedSums[this.idx]
            , data.activations[this.idx]);
    }

    calcOutputNodeValues(data: DataPointRunData, correctResults: number[]) {
        const costDerivatives = Cost.derivative(
            data.activations[this.idx],
            correctResults
        );
        const activationDerivatives = this.activation.derivative(
            data.weightedSums[this.idx]
        );

        for (let i = 0; i < this.nodes; i++) {
            data.nodeValues[this.idx][i] = costDerivatives[i] * activationDerivatives[i];
        }

        // data.nodeValues[this.idx] = costDerivatives.map(
        //     (c, i) => c * activationDerivatives[i]
        // );
    }

    calcHiddenLayerNodeValues(data: DataPointRunData, nextLayer: Layer) {
        const activationDerivatives = this.activation.derivative(
            data.weightedSums[this.idx]
        );

        for (let node = 0; node < this.nodes; node++) {
            let sum = 0;
            for (let output = 0; output < nextLayer.nodes; output++) {
                sum += nextLayer.getWeight(output, node) * data.nodeValues[this.idx + 1][output]
            }

            data.nodeValues[this.idx][node] = activationDerivatives[node] * sum;
        }

        // data.nodeValues[this.idx] = activationDerivatives.map(
        //     (activationDerivative, node) => {
        //         return (
        //             activationDerivative *
        //             data.nodeValues[this.idx + 1].reduce(
        //                 (prev, outputNodeValue, output) =>
        //                     prev +
        //                     nextLayer.getWeight(output, node) * outputNodeValue, 0
        //             )
        //         );
        //     }
        // );
    }

    calcGradients(data: DataPointRunData) {
        let inputs = this.idx == 0 ? data.inputs : data.activations[this.idx - 1];

        for (let node = 0; node < this.nodes; node++) {
            for (let input = 0; input < this.inputs; input++) {
                this.weightGradients[this.getWeightIdx(node, input)] += data.nodeValues[this.idx][node] * inputs[input];
            }

            this.biasGradients[node] += data.nodeValues[this.idx][node];
        }
        // data.nodeValues[this.idx].forEach((nodeValue, node) => {
        //     inputs.forEach((inputValue, input) => {
        //         this.weightGradients[this.getWeightIdx(node, input)] +=
        //             nodeValue * inputValue;
        //     });
        //     this.biasGradients[node] += nodeValue;
        // });
    }

    applyGradients(
        learnRate: number,
        regularization: number,
        momentum: number
    ) {
        const weightDecay = 1 - regularization * learnRate;

        for (let i = 0; i < this.nodes * this.inputs; i++) {

            this.weightVelocities[i] =
                this.weightVelocities[i] * momentum -
                this.weightGradients[i] * learnRate;
            this.weights[i] =
                this.weights[i] * weightDecay + this.weightVelocities[i];
            this.weightGradients[i] = 0;
        }

        // this.weightGradients.forEach((weightCost, i) => {
        //     this.weightVelocities[i] =
        //         this.weightVelocities[i] * momentum -
        //         this.weightGradients[i] * learnRate;
        //     this.weights[i] =
        //         this.weights[i] * weightDecay + this.weightVelocities[i];
        //     this.weightGradients[i] = 0;
        // });

        for (let i = 0; i < this.nodes; i++) {

            this.biasVelocities[i] =
                this.biasVelocities[i] * momentum -
                this.biasGradients[i] * learnRate;
            this.biases[i] += this.biasVelocities[i];

            this.biasGradients[i] = 0;
        }

        // this.biasGradients.forEach((biasCost, i) => {
        //     this.biasVelocities[i] =
        //         this.biasVelocities[i] * momentum -
        //         this.biasGradients[i] * learnRate;
        //     this.biases[i] += this.biasVelocities[i];

        //     this.biasGradients[i] = 0;
        // });
    }

    reset() {
        this.weightGradients = new Array(this.nodes * this.inputs).fill(0);
        this.biasGradients = new Array(this.nodes).fill(0);
    }
}
