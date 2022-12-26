export abstract class Activation {
    abstract function(weightedSums: number[]): number[];
    abstract derivative(weightedSums: number[]): number[];

    static Relu: Activation;
    static Sigmod: Activation;
    static SoftMax: Activation;
}

abstract class SimpleActivation extends Activation {
    abstract singleFunction(weightedSum: number): number;
    abstract singleDerivative(weightedSum: number): number;

    function(weightedSums: number[]): number[] {
        return weightedSums.map((n) => this.singleFunction(n));
    }

    derivative(weightedSums: number[]): number[] {
        return weightedSums.map((n) => this.singleDerivative(n));
    }
}

export class Relu extends SimpleActivation {
    singleFunction(weightedSum: number): number {
        return Math.max(weightedSum, 0);
    }

    singleDerivative(weightedSum: number): number {
        return weightedSum > 0 ? 1 : 0;
    }
}

export class Sigmoid extends SimpleActivation {
    singleFunction(weightedSum: number): number {
        return 1 / (1 + Math.exp(-weightedSum));
    }

    singleDerivative(weightedSum: number): number {
        let a = this.singleFunction(weightedSum);
        return a * (1 - a);
    }
}

export class SoftMax extends Activation {
    function(weightedSums: number[]): number[] {
        const expSum = weightedSums.reduce(
            (prev, curr) => prev + Math.exp(curr),
            0
        );

        return weightedSums.map((n) => Math.exp(n) / expSum);
    }

    derivative(weightedSums: number[]): number[] {
        const expSum = weightedSums.reduce(
            (prev, curr) => prev + Math.exp(curr),
            0
        );
        return weightedSums.map((n) => {
            const exp = Math.exp(n);
            return (exp * expSum - exp * exp) / (expSum * expSum);
        });
    }
}

Activation.Relu = new Relu();
Activation.Sigmod = new Sigmoid();
Activation.SoftMax = new SoftMax();
//@ts-ignore

window.activation = Activation;
