import { Activation } from "./activation.js";
import { Cost } from "./cost.js";
import { arrFromFunc, randomInNormalDistribution } from "./utils.js";

export class Layer {
    weights: number[];
    biases: number[];

    weightGradients: number[];
    biasGradients: number[];
    nodeValues: number[];

    weightVelocities: number[];
    biasVelocities: number[];

    prevRun: {
        inputs: number[];
        weightedSums: number[];
        activations: number[];
    } = {
        inputs: [],
        weightedSums: [],
        activations: [],
    };
    id: number;

    static nextId = 0;

    constructor(
        public nodes: number,
        public inputs: number,
        public activation: Activation
    ) {
        this.id = Layer.nextId++;
        this.biases = arrFromFunc(nodes, () => Math.random() - 0.5);

        const sqrt = Math.sqrt(inputs);
        this.weights = arrFromFunc(nodes * inputs, () => {
            let y = randomInNormalDistribution(1, 0);
            return y / sqrt;
        });

        this.nodeValues = new Array(nodes).fill(0);
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

    feedForward(inputs: number[]) {
        this.prevRun.inputs = inputs.slice();

        this.prevRun.weightedSums = this.biases.map((bias, node) =>
            inputs.reduce(
                (prev, curr, input) =>
                    prev + curr * this.getWeight(node, input),
                bias
            )
        );

        this.prevRun.activations = this.activation.function(
            this.prevRun.weightedSums
        );

        return this.prevRun.activations;
    }

    calcOutputNodeValues(correctResults: number[], cost: Cost) {
        const costDerivatives = cost.derivative(
            this.prevRun.activations,
            correctResults
        );
        const activationDerivatives = this.activation.derivative(
            this.prevRun.weightedSums
        );
        this.nodeValues = costDerivatives.map(
            (c, i) => c * activationDerivatives[i]
        );
    }

    calcHiddenLayerNodeValues(nextLayer: Layer) {
        const activationDerivatives = this.activation.derivative(
            this.prevRun.weightedSums
        );

        this.nodeValues = activationDerivatives.map(
            (activationDerivative, node) => {
                return (
                    activationDerivative *
                    nextLayer.nodeValues.reduce(
                        (prev, outputNodeValue, output) =>
                            prev +
                            nextLayer.getWeight(output, node) * outputNodeValue
                    )
                );
            }
        );
    }

    calcGradients() {
        this.nodeValues.forEach((nodeValue, node) => {
            this.prevRun.inputs.forEach((inputValue, input) => {
                this.weightGradients[this.getWeightIdx(node, input)] +=
                    nodeValue * inputValue;
            });
            this.biasGradients[node] += nodeValue;
        });
    }

    applyGradients(
        learnRate: number,
        regularization: number,
        momentum: number
    ) {
        const weightDecay = 1 - regularization * learnRate;

        this.weightGradients.forEach((weightCost, i) => {
            this.weightVelocities[i] =
                this.weightVelocities[i] * momentum -
                this.weightGradients[i] * learnRate;
            this.weights[i] =
                this.weights[i] * weightDecay + this.weightVelocities[i];
            this.weightGradients[i] = 0;
        });

        this.biasGradients.forEach((biasCost, i) => {
            this.biasVelocities[i] =
                this.biasVelocities[i] * momentum -
                this.biasGradients[i] * learnRate;
            this.biases[i] += this.biasVelocities[i];

            this.biasGradients[i] = 0;
            this.nodeValues[i] = 0;
        });
    }
}
