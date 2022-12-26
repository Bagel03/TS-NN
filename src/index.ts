import { Network } from "./network.js";
import { flip } from "./utils.js";

const NN = new Network(2, 4, 2, 2);
const data: [inputs: number[], correctOutputs: number[]][] = [
    // [
    //     [0, 0],
    //     [0, 1],
    // ],
    // [
    //     [0, 1],
    //     [1, 0],
    // ],
    // [
    //     [1, 0],
    //     [1, 0],
    // ],
    // [
    //     [1, 1],
    //     [0, 1],
    // ],
];

let z = 5;
// const isValid = (a: number, b: number) => z ** a + z ** b < z + 1;
// const isValid = (a: number, b: number) =>
//     (1.5 * a - 0.75) ** 3 + (0.01 * a) ** 2 + 0.5 > b;
const isValid = (a: number, b: number) => Math.cos(2 * Math.PI * a) / 2 + 0.4 > b;

for (let i = 0; i < 1000; i++) {
    let a = Math.random();
    let b = Math.random();
    let isValidBit: 0 | 1 = isValid(a, b) ? 1 : 0;
    data.push([
        [a, b],
        [isValidBit, flip(isValidBit)],
    ]);
}

const canvas = document.getElementsByTagName("canvas")[0];
const context = canvas.getContext("2d")!;
const costSpan = document.getElementById("cost")!;
const accuracySpan = document.getElementById("accuracy")!;

const rectSize = 5;
const numRects = 50;
context.canvas.width = rectSize * numRects * 2 + rectSize;
context.canvas.height = rectSize * numRects;

// Draw the correct thing:
context.translate(rectSize * (numRects + 1), 0);
for (let y = 0; y < numRects; y++) {
    for (let x = 0; x < numRects; x++) {
        const t = isValid(x / numRects, y / numRects);
        context.fillStyle = t ? "blue" : "red";
        context.fillRect(x * rectSize, y * rectSize, rectSize, rectSize);
    }
}
context.resetTransform();

let lastCost = NN.totalCost(data);
let lastAccuracy = 0;

let i = 0;
let learnRate = 1;
let learnRateDecay = 0; // 0.00075;
let run = 0;
const render = () => {
    const { accuracy, averageCost } = NN.train(data, (learnRate *= 1 - learnRateDecay));

    for (let y = 0; y < numRects; y++) {
        for (let x = 0; x < numRects; x++) {
            const [t] = NN.feedForward([x / numRects, y / numRects]).activations.at(-1);
            // const t = isValid(x / numRects, y / numRects) ? 1 : 0;
            const value = ((0xff * t) | 0).toString(16);
            context.fillStyle = `#${value}${value}${value}`;
            context.fillRect(x * rectSize, y * rectSize, rectSize, rectSize);
        }
    }

    const newCost = averageCost
    if (newCost < lastCost) {
        costSpan.style.color = "green";
    } else {
        costSpan.style.color = "red";
    }
    costSpan.innerHTML = newCost.toFixed(10);
    lastCost = newCost;

    if (accuracy >= lastAccuracy) {
        accuracySpan.style.color = "green";
    } else {
        accuracySpan.style.color = "red";
    }
    accuracySpan.innerHTML = accuracy.toFixed(4) + `\t(Run: ${run})`;
    lastAccuracy = accuracy;
    run++;
    // const results = data.map((dataPoint) => NN.feedForward(dataPoint[0]));

    // results.forEach(([t, f], i) => {
    //     const x = i % 2;
    //     const y = Math.floor(i / 2);

    //     const value = ((0xff * t) | 0).toString(16);
    //     context.fillStyle = `#${value}${value}${value}`;
    //     context.fillRect(x * rectSize, y * rectSize, rectSize, rectSize);
    // });
};

// console.log(JSON.parse(JSON.stringify(NN)));
// // console.log(NN.feedForward(data[0][0]));
// for (let i = 0; i < 10; i++) {
//     NN.train(data, 1);
// }
// console.log(NN);
// console.log(NN.feedForward(data[0][0]));

const startRender = () => {
    i = window.setInterval(() => {
        render();
    }, 1000 / 100);
};

// window.setInterval(() => {
//     console.clear();
//     console.log(learnRate);
// }, 1 * 1000);

startRender();

document.getElementsByTagName("button")[0].onclick = () => {
    // render();
    // console.log(JSON.parse(JSON.stringify(NN)));
    // NN.layers.forEach(l => l.reset());

    if (!i) return startRender();

    console.log(NN.totalCost(data));
    console.log(data.map((dataPoint) => NN.feedForward(dataPoint[0])));
    window.clearInterval(i);
    i = 0;
};

//@ts-ignore
window.nn = NN;
