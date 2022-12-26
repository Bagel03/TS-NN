import { Network } from "./network";

export class DataPointRunData {
    activations: number[][] = [];
    weightedSums: number[][] = [];
    nodeValues: number[][] = [];

    constructor(public inputs: number[], network: Network) {
        for (let i = 0; i < network.layerSizes.length; i++) {
            this.activations.push(new Array(network.layerSizes[i]).fill(0));
            this.weightedSums.push(new Array(network.layerSizes[i]).fill(0));
            this.nodeValues.push(new Array(network.layerSizes[i]).fill(0));
        }

    }
}