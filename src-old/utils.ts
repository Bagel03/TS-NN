export const arrFromFunc = <T>(
    len: number,
    func: (index: number) => T
): Array<T> => {
    let a = [];
    for (let i = 0; i < len; i++) {
        a[i] = func(i);
        // console.log(a[i]);
    }
    return a;
};

export const randomInNormalDistribution = (sd: number, mean: number) => {
    const x = Math.random();
    const y = Math.random();
    const z = Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y);
    return z * sd + mean;
};

export const flip = (bit: 0 | 1) => {
    return Math.abs(bit - 1);
};

export const maxIdx = (arr: number[]) => {
    return arr.reduce(
        (prev, num, i) => {
            if (num > prev.max) {
                return { max: num, idx: i };
            } else {
                return prev;
            }
        },
        { max: -Infinity, idx: -1 }
    ).idx;
};

//@ts-ignore
window.maxIdx = maxIdx;
