let model;

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    return model;
}

function generateData() {
    const xs = tf.tensor2d([
        3.1416, 2.8274, 2.5133, 2.1991, 1.8850, 
        1.5708, 1.2566, 0.9425, 0.6283, 0.3142
    ], [10, 1]);
    const ys = tf.tensor2d([
        10.5664, 9.3096, 8.0532, 6.7964, 5.5400, 
        4.2832, 3.0264, 1.7700, 0.5132, -0.7432
    ], [10, 1]);
    return {xs, ys};
}

async function trainModel(model, xs, ys) {
    await model.fit(xs, ys, {
        epochs: 10000,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (epoch % 1000 === 0) {
                    console.log(`Epoch: ${epoch}, Loss: ${logs.loss}`);
                }
            }
        }
    });
}

function makePrediction(model, input) {
    const inputTensor = tf.tensor2d([input], [1, 1]);
    const outputTensor = model.predict(inputTensor);
    const output = outputTensor.dataSync();
    return output[0];
}

async function main() {
    model = createModel();
    const {xs, ys} = generateData();
    await trainModel(model, xs, ys);
    document.getElementById('output').innerText = makePrediction(model, 4);
}

async function predict() {
    const inputX = parseFloat(document.getElementById('inputX').value);
    document.getElementById('inputValue').innerText = inputX;
    const prediction = makePrediction(model, inputX);
    document.getElementById('output').innerText = prediction;
}

main();
