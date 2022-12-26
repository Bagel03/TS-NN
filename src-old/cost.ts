export abstract class Cost {
    abstract function(
        generatedResults: number[],
        correctResults: number[]
    ): number;
    abstract derivative(
        generatedResults: number[],
        correctResults: number[]
    ): number[];

    static MSE: Cost;
    static CrossEntropy: Cost;
}

class CrossEntropy extends Cost {
    function(generatedResults: number[], correctResults: number[]): number {
        return -generatedResults.reduce((prev, generated, i) => {
            const correct = correctResults[i];

            const v =
                correct == 1 ? Math.log(generated) : Math.log(1 - generated);
            return prev + (Number.isNaN(v) ? 0 : v);
        }, 0);
    }

    derivative(generatedResults: number[], correctResults: number[]): number[] {
        return generatedResults.map((generated, i) => {
            const correct = correctResults[i];
            if (generated == 0 || generated == 1) {
                console.log("solved")
                return 0;

            }
            return (correct - generated) / (generated * (generated - 1));
        });
    }
}

Cost.CrossEntropy = new CrossEntropy();
//@ts-ignore
window.cost = Cost;
