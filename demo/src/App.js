import React from 'react';
import { Typography } from 'antd';
import './App.css';
import ModelShowcase from './components/ModelShowcase';
import InferenceShowcase from './components/InferenceShowcase';
import CustomNet from './models/CustomNet';
const { Text, Paragraph, Link } = Typography;

function App() {
  const p = process.env.PUBLIC_URL;

  return (
    <article className="App">
      <header className="App-header">
        <h1>Benchmarking Optimizers for Sign Language detection</h1>
        <h4>
          <Text type="secondary">Using Deep Learning with Keras</Text>
        </h4>
      </header>
      <Paragraph>Hey! Welcome to a live demonstration page of how our trained network performs. We trained a network on the ASL sign language <Link href='https://kaggle.com/grassknoted/asl-alphabet'>dataset</Link>, aiming to differentiate between 29 classes. We demonstrate the network trained using the <b>Adam</b> optimizer, which yielded reasonable validation classification performance; about 90% accuracy. A learning rate of 0.001 was used, all other hyperparameters were standard. Let's see how well it performs, in an interactive way. </Paragraph>
      <ModelShowcase modelFile={p+'/adam/AdamOptimizer-NN.json'} model={CustomNet}>
        <Paragraph>
          Let's first test the model on images it has seen before, training images. It should be able to get these predicted correctly relatively easily.
        </Paragraph>

        <InferenceShowcase pictureUrls={[
            p+'/asl_alphabet_test/A_1.jpg',
            p+'/asl_alphabet_test/B_1.jpg',
            p+'/asl_alphabet_test/C_1.jpg'
          ]}/>

        <Paragraph>Next, we can predict images using <b>unseen</b> data, test data. </Paragraph>

        <InferenceShowcase pictureUrls={[
            p+'/asl_alphabet_test/E_test.jpg',
            p+'/asl_alphabet_test/F_test.jpg',
            p+'/asl_alphabet_test/G_test.jpg',
            p+'/asl_alphabet_test/H_test.jpg'
          ]}/>
        <Paragraph>Or optionally: upload your own images to predict! Try to make a clear photo and see whether our network is able to predict correctly.</Paragraph>

        <InferenceShowcase/>
      </ModelShowcase>

      <Paragraph>This project uses <Link href='https://www.tensorflow.org/js'>TensorFlow.js</Link> to make live inferences in the browser. Our trained Keras model was converted using the <Text code>tfjs-converter</Text>, and then loaded up into this React.js application.</Paragraph>
      
      <Paragraph>Project built as part of the Deep Learning course <Text code>WMCS001-05</Text> taught at the University of Groningen. </Paragraph>
      <Paragraph>
        <small>
          <Text type='secondary'>
            &gt; All our code is available on&nbsp;
            <Link href='https://github.com/dunnkers/deep-learning/'>Github <img src={p+'/github32.png'} 
              alt='Github logo'
              style={{width: 16, verticalAlign: 'text-bottom'}} />
            </Link>
          </Text>
        </small>
      </Paragraph>
    </article>
  );
}

export default App;
