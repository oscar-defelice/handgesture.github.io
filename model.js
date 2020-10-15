let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new HandDataset();
var zeroSamples=0, oneSamples=0, twoSamples=0, threeSamples=0, fourSamples=0, fiveSamples=0;
let isPredicting = false;
let numClasses = 6

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(numClasses);
    
  // In the space below create a neural network that can classify hand gestures
  // corresponding to zero, one, two, three, four and five. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 6 classes, your output
  // layer should have 6 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.  
  model = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({units: 128, activation: 'relu'}),
        tf.layers.dense({units: 64, activation: 'relu'}),
        tf.layers.dense({units: numClasses, activation: 'softmax'})
    ]
  });
    
   
  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001)
    
        
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(6);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			zeroSamples++;
			document.getElementById("zeroSamples").innerText = "Zero samples:" + zeroSamples;
			break;
		case "1":
			oneSamples++;
			document.getElementById("oneSamples").innerText = "One samples:" + oneSamples;
			break;
		case "2":
			twoSamples++;
			document.getElementById("twoSamples").innerText = "Two samples:" + twoSamples;
			break;  
		case "3":
			threeSamples++;
			document.getElementById("threeSamples").innerText = "Three samples:" + threeSamples;
			break;
        case "4":
            fourSamples++;
            document.getElementById("fourSamples").innerText = "Four samples" + fourSamples;
            break;
        case "5":
            fiveSamples++;
            document.getElementById("fiveSamples").innerText = "Five samples" + fiveSamples;
            break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Zero";
			break;
		case 1:
			predictionText = "I see One";
			break;
		case 2:
			predictionText = "I see Two";
			break;
		case 3:
			predictionText = "I see Three";
			break;
        case 4:
			predictionText = "I see Four";
			break;
        case 5:
			predictionText = "I see Five";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


function saveModel(){
    model.save('downloads://my_model');
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();