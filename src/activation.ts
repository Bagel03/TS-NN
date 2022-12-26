export abstract class Activation {
    abstract function(weightedSums: number[], into: number[]);
    abstract derivative(weightedSums: number[]): number[];

    static Relu: Activation;
    static Sigmod: Activation;
    static SoftMax: Activation;
}

// abstract class SimpleActivation extends Activation {
//     abstract singleFunction(weightedSum: number): number;
//     abstract singleDerivative(weightedSum: number): number;

//     function(weightedSums: number[], into: number[]) {
//         return weightedSums.map((n) => this.singleFunction(n));
//     }

//     derivative(weightedSums: number[]): number[] {
//         return weightedSums.map((n) => this.singleDerivative(n));
//     }
// }

export class Relu extends Activation {
    function(weightedSums: number[], into: number[]) {
        for (let i = 0; i < weightedSums.length; i++) {
            into[i] = Math.max(weightedSums[i], 0);
        }
    }

    derivative(weightedSums: number[]): number[] {
        let result = new Array(weightedSums.length).fill(0);
        for (let i = 0; i < weightedSums.length; i++) {
            if (weightedSums[i] > 0) result[i] = 1;
        }
        return result
    }
}

export class Sigmoid extends Activation {
    function(weightedSums: number[], into: number[]) {
        for (let i = 0; i < weightedSums.length; i++) {
            into[i] = 1 / (1 + Math.exp(-weightedSums[i]))
        }
    }

    derivative(weightedSums: number[]): number[] {
        let result = new Array(weightedSums.length).fill(0);
        for (let i = 0; i < weightedSums.length; i++) {
            let a = 1 / (1 + Math.exp(-weightedSums[i]));
            result[i] = a * (1 - a);
        }
        return result;
    }
}

export class SoftMax extends Activation {
    function(weightedSums: number[], into: number[]) {
        let sum = 0;
        for (let i = 0; i < weightedSums.length; i++) {
            sum += Math.exp(weightedSums[i]);
        }

        for (let i = 0; i < weightedSums.length; i++) {
            into[i] = Math.exp(weightedSums[i]) / sum;
        }
    }

    derivative(weightedSums: number[]): number[] {
        let result = new Array(weightedSums.length).fill(0);

        let sum = 0;
        for (let i = 0; i < weightedSums.length; i++) {
            sum += Math.exp(weightedSums[i]);
        }

        for (let i = 0; i < weightedSums.length; i++) {
            const exp = Math.exp(weightedSums[i]);
            result[i] = (exp * sum - exp * exp) / (sum * sum);
        }

        return result;
        // return weightedSums.map((n) => {
        //     const exp = Math.exp(n);
        //     return (exp * expSum - exp * exp) / (expSum * expSum);
        // });
    }
}

Activation.Relu = new Relu();
Activation.Sigmod = new Sigmoid();
Activation.SoftMax = new SoftMax();
//@ts-ignore

window.activation = Activation;
