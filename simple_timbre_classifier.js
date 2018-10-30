const max = require('max-api');
const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');

let model;
let xs, ys;
let isTraining = false;
let isReady = false;

let epochs = 5000;
let lr = 0.25;

let labelList = [
  'I hear nothing...',
  'This sounds soft...',
  'This sounds dark...',
  'This sounds rough...',
  'This sounds bright...',
  'This is LOUD!'
]

//making up training data
xs = tf.tensor2d([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], //nothing
  [0.6, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], //dark
  [0.3, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], //soft
  [0.8, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1], //loud
  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2], //rough
  [0, 0, 0, 0.1, 0.2, 0.4, 0.5, 0.5, 0.3, 0.5, 0.6, 0.5, 0.6, 0.5, 0.5, 0.4], //bright
  [0.3, 0.2, 0.2, 0.4, 0.3, 0.4, 0.4, 0.1, 0.3, 1, 0.2, 0.4, 0.3, 0.3, 0.4, 0.1], //rough
  [0.1, 0.7, 0.4, 0.3, 0.2, 1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 1, 0.1, 0.1], //soft
  [0.8, 0.2, 0.3, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], //dark
  [0.1, 0.1, 0.2, 0.2, 0.4, 0.7, 0.6, 0.7, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2], //bright
  [0.8, 0.6, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5] //loud

]);

//label (one-hot)
ys = tf.tensor2d([
  [1, 0, 0, 0, 0, 0], //nothing
  [0, 0, 1, 0, 0, 0], //dark
  [0, 1, 0, 0, 0, 0], //soft
  [0, 0, 0, 0, 0, 1], //loud
  [0, 0, 0, 1, 0, 0], //rough
  [0, 0, 0, 0, 1, 0], //bright
  [0, 0, 0, 1, 0, 0], //rough
  [0, 1, 0, 0, 0, 0], //soft
  [0, 0, 1, 0, 0, 0], //dark
  [0, 0, 0, 0, 1, 0], //bright
  [0, 0, 0, 0, 0, 1] //loud
]);

model = buildModel();

max.outlet('report', 'Ready... Click train!');

//There should be a way to combine these handlers function into one :p
max.addHandler('epoch', (epoch) => {
  epochs = epoch;
})

max.addHandler('learningrate', (learningrate) => {
  lr = learningrate;
})

max.addHandler('train', () => {
  train();
});

//METHOD FOR PREDICTING
max.addHandler('predict', (...predict) => {
  if (!isTraining && isReady) {
    tf.tidy(() => {
      max.post(...predict);
      const input = tf.tensor2d([predict]);
      let results = model.predict(input);
      let argMax = results.argMax(1);
      let index = argMax.dataSync()[0];
      let label = labelList[index];
      max.outlet('report', label);
    })
  }
});

//METHOD FOR TRAINING
async function train() {
  if (isTraining) {
    return;
  }
  isTraining = true;
  let loss;
  let acc;
  await model.fit(xs, ys, {
    shuffle: true,
    validationSplit: 0.1,
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        loss = logs.loss.toFixed(8);
        acc = logs.acc.toFixed(8);
        max.outlet('report', `
        epoch ${epoch} of ${epochs}
        loss: ${loss}
        acc: ${acc}
        `);
      },
      onTrainEnd: () => {
        isTraining = false;
        isReady = true;
        max.outlet('report', `
        Training finished with ${loss} loss and ${acc} accuracy
        `)
      }
    }
  })
}

//METHOD FOR BUILDING THE MODEL
function buildModel() {
  let md = tf.sequential();
  const hidden1 = tf.layers.dense({
    units: 32,
    inputShape: [16],
    useBias: true,
    activation: 'relu'
  });

  const hidden2 = tf.layers.dense({
    units: 64,
    useBias: true,
    activation: 'sigmoid'
  });

  const output = tf.layers.dense({
    units: 6,
    activation: 'softmax'
  });
  md.add(hidden1);
  md.add(hidden2);
  md.add(output);

  const optimizer = tf.train.sgd(lr);

  md.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return md;
}