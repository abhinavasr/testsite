let net;
// Get webcam control


// Get KNN control
const classifier = loadClassifierFromLocalStorage(); // knnClassifier.create()

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = username => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const webcamElement = document.getElementById('webcam');
    const activation = net.infer(webcamElement, 'conv_preds');
    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, username);
    saveModel();
  };

  async function saveModel(){
    saveClassifierInLocalStorage(classifier);
  }

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(document.getElementById('username').value));

  const webcamElement = document.getElementById('webcam');

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      // Print confidence
      console.log(`
      prediction: ${result.label}\n
      probability: ${result.confidences[result.label]}
      `);
    }
    await tf.nextFrame();
  }
}

// setup webcam
async function setupWebcam() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        const webcamElement = document.getElementById('webcam');
        navigator.getUserMedia({video: true},
          stream => {
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata',  () => resolve(), false);
          },
          error => reject());
      } else {
        reject();
      }
    });
  }

app();

// Saving and Loading Classifier
async function saveClassifierInLocalStorage(classifier) {
  let dataset = classifier.getClassifierDataset()
   var datasetObj = {}
   Object.keys(dataset).forEach((key) => {
     let data = dataset[key].dataSync();
     datasetObj[key] = Array.from(data); 
   });
   let jsonStr = JSON.stringify(datasetObj)
   localStorage.setItem("face_data", jsonStr);
}

function loadClassifierFromLocalStorage() {
  const classifier = knnClassifier.create()
  //can be change to other source
  let dataset = localStorage.getItem("face_data")
  if(dataset!=null){
    let tensorObj = JSON.parse(dataset)
    //covert back to tensor
    Object.keys(tensorObj).forEach((key) => {
      tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024])
    })
    classifier.setClassifierDataset(tensorObj)
  }
  return classifier;
}