// export abstract class Cost {
//     abstract function(
//         generatedResults: number[],
//         correctResults: number[]
//     ): number;
//     abstract derivative(
//         generatedResults: number[],
//         correctResults: number[]
//     ): number[];

//     static MSE: Cost;
//     static CrossEntropy: Cost;
// }

export class Cost {
    static function(generatedResults: number[], correctResults: number[]) {
        let sum = 0;

        for (let i = 0; i < generatedResults.length; i++) {
            const v = correctResults[i] == 1 ? Math.log(generatedResults[i]) : Math.log(1 - generatedResults[i]);

            if (!Number.isNaN(v)) { sum += v }
        }

        return -sum;

    }

    // static function(generatedResults: number[], correctResults: number[]): number {
    //     return -generatedResults.reduce((prev, generated, i) => {
    //         const correct = correctResults[i];

    //         const v =
    //             correct == 1 ? Math.log(generated) : Math.log(1 - generated);
    //         return prev + (Number.isNaN(v) ? 0 : v);
    //     }, 0);
    // }

    static derivative(generatedResults: number[], correctResults: number[]): number[] {
        const result = new Array(generatedResults.length).fill(0);

        for (let i = 0; i < generatedResults.length; i++) {
            if (generatedResults[i] == 0 || generatedResults[i] == 1) continue;
            const correct = correctResults[i];
            const generated = generatedResults[i];

            result[i] = (correct - generated) / (generated * (generated - 1));
        }

        return result;
    }

    // static _derivative(generatedResults: number[], correctResults: number[]): number[] {
    //     return generatedResults.map((generated, i) => {
    //         const correct = correctResults[i];

    //         // console.log(correct, generated)
    //         if (generated == 0 || generated == 1) {
    //             console.log("solved")
    //             return 0;

    //         }
    //         return (correct - generated) / (generated * (generated - 1));
    //     });
    // }
}


