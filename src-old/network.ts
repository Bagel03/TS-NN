import { Activation } from "./activation.js";
import { Cost } from "./cost.js";
import { Layer } from "./layer.js";
import { maxIdx } from "./utils.js";

export class Network {
    readonly layers: Layer[];
    readonly layerSizes: number[];
    readonly cost: Cost = Cost.CrossEntropy;

    constructor(inputs: number, ...hiddenLayerSizes: number[]) {
        this.layerSizes = hiddenLayerSizes;

        this.layers = hiddenLayerSizes.map(
            (nodes, i) =>
                new Layer(
                    nodes,
                    i == 0 ? inputs : hiddenLayerSizes[i - 1],
                    i == hiddenLayerSizes.length - 1
                        ? Activation.SoftMax
                        : Activation.Sigmod
                )
        );
    }

    get outputLayerIndex() {
        return this.layers.length - 1;
    }

    get outputLayer() {
        return this.layers[this.outputLayerIndex];
    }

    feedForward(inputs: number[]) {
        for (const layer of this.layers) {
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }

    // trains, returns if guess was correct
    trainOnSinglePoint(inputs: number[], correctOutputs: number[]): boolean {
        const guess = maxIdx(this.feedForward(inputs));

        this.outputLayer.calcOutputNodeValues(correctOutputs, this.cost);
        this.outputLayer.calcGradients();

        for (let i = this.outputLayerIndex - 1; i >= 0; i--) {
            this.layers[i].calcHiddenLayerNodeValues(this.layers[i + 1]);
            this.layers[i].calcGradients();
        }
        return guess == maxIdx(correctOutputs);
    }

    train(
        dataPoints: [inputs: number[], correctOutputs: number[]][],
        learnRate: number
    ) {
        let correct = 0;
        for (const [inputs, correctOutputs] of dataPoints) {
            if (this.trainOnSinglePoint(inputs, correctOutputs)) correct++;
        }

        this.layers.forEach((layer) =>
            layer.applyGradients(learnRate / dataPoints.length, 0.1, 0.9)
        );
        return correct / dataPoints.length;
    }

    totalCost(dataPoints: [inputs: number[], correctOutputs: number[]][]) {
        const results = dataPoints.map((point) => this.feedForward(point[0]));

        const totalCost = results.reduce(
            (prev, curr, i) =>
                prev + this.cost.function(curr, dataPoints[i][1]),
            0
        );
        return totalCost / dataPoints.length;
    }
}
