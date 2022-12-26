import { Activation } from "./activation.js";
import { Cost } from "./cost.js";
import { DataPointRunData } from "./data.js";
import { Layer } from "./layer.js";
import { maxIdx } from "./utils.js";

export class Network {
    readonly layers: Layer[];
    readonly layerSizes: number[];

    constructor(...layerSizes: number[]) {
        this.layerSizes = layerSizes.slice(1);

        this.layers = [];
        for (let i = 1; i < layerSizes.length; i++) {
            let activation = i == layerSizes.length - 1 ? Activation.SoftMax : Activation.Sigmod;
            this.layers.push(new Layer(layerSizes[i], layerSizes[i - 1], activation, i - 1))
        }
        console.log("Made")
        // this.layers = hiddenLayerSizes.map(
        //     (nodes, i) =>
        //         new Layer(
        //             nodes,
        //             i == 0 ? inputs : hiddenLayerSizes[i - 1],
        //             i == hiddenLayerSizes.length - 1
        //                 ? Activation.SoftMax
        //                 : Activation.Sigmod,
        //             i
        //         )
        // );
    }


    feedForward(inputs: number[]) {
        let data = new DataPointRunData(inputs, this);
        for (const layer of this.layers) {
            layer.feedForward(data);
        }
        return data;
    }

    // trains, returns if guess was correct
    trainOnSinglePoint(inputs: number[], correctOutputs: number[]): boolean {
        let data = this.feedForward(inputs)
        const guess = maxIdx(data.activations[data.activations.length - 1]);

        this.layers[this.layers.length - 1].calcOutputNodeValues(data, correctOutputs);
        this.layers[this.layers.length - 1].calcGradients(data);

        for (let i = this.layers.length - 2; i >= 0; i--) {
            this.layers[i].calcHiddenLayerNodeValues(data, this.layers[i + 1]);
            this.layers[i].calcGradients(data);
        }

        return guess == maxIdx(correctOutputs);
    }

    train(
        dataPoints: [inputs: number[], correctOutputs: number[]][],
        learnRate: number,
        storeData = true
    ) {

        let correct = 0;
        let totalCost = 0;
        for (const [inputs, correctOutputs] of dataPoints) {
            let data = this.feedForward(inputs)

            this.layers[this.layers.length - 1].calcOutputNodeValues(data, correctOutputs);
            this.layers[this.layers.length - 1].calcGradients(data);

            for (let i = this.layers.length - 2; i >= 0; i--) {
                this.layers[i].calcHiddenLayerNodeValues(data, this.layers[i + 1]);
                this.layers[i].calcGradients(data);
            }

            if (!storeData) continue;
            const guess = maxIdx(data.activations[data.activations.length - 1]);
            if (guess == maxIdx(correctOutputs)) correct++;
            totalCost += Cost.function(data.activations[data.activations.length - 1], correctOutputs);

            // if (this.trainOnSinglePoint(inputs, correctOutputs)) correct++;


        }

        for (const layer of this.layers) {
            layer.applyGradients(learnRate / dataPoints.length, 0.1, 0.9)

        }


        return {
            accuracy: correct / dataPoints.length,
            averageCost: totalCost / dataPoints.length
        }
    }

    totalCost(dataPoints: [inputs: number[], correctOutputs: number[]][]) {
        const results = dataPoints.map((point) => this.feedForward(point[0]));

        const totalCost = results.reduce(
            (prev, curr, i) =>
                prev + Cost.function(curr.activations[curr.activations.length - 1], dataPoints[i][1]),
            0
        );
        return totalCost / dataPoints.length;
    }
}
