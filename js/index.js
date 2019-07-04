let net;
// Get KNN control
const classifier = loadClassifierFromLocalStorage(); // knnClassifier.create()

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');
  
  // Wait for cam to be there
  await setupWebcam();
  const webcamElement = document.getElementById('webcam');

  document.getElementById('console').addEventListener('click', () => userLogin(document.getElementById('hiddenAuthData').value));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      // Print confidence
      // console.log(`
      // prediction: ${result.label}\n
      // probability: ${result.confidences[result.label]}
      // `);

      if(result.confidences[result.label] > 0.90){
        document.getElementById('console').innerText = "Login as "+result.label;
        document.getElementById('hiddenAuthData').value = result.label;
      }else{
        document.getElementById('console').innerText = "Unrecognized User (Login with Username/Password)";
        document.getElementById('hiddenAuthData').value = "";
      }
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

// Load classifier
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

function userLogin(textContent){
  if(textContent == ""){
      alert("Please register to proceed");
  }else{
    alert("Welcome "+textContent);
  }
}