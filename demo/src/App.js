import React from 'react';
import { Typography } from 'antd';
import './App.css';
import ModelShowcase from './components/ModelShowcase';
import InferenceShowcase from './components/InferenceShowcase';
import CustomNet from './models/CustomNet';
const { Paragraph } = Typography;

function App() {
  const p = process.env.PUBLIC_URL;

  return (
    <article className="App">
      <ModelShowcase modelFile={p+'/adam/AdamOptimizer-NN.json'} model={CustomNet}>
        <Paragraph>
          Model.
        </Paragraph>

        <InferenceShowcase pictureUrls={[
            p+'/asl_alphabet_test/A1.jpg',
            p+'/asl_alphabet_test/B1.jpg',
            p+'/asl_alphabet_test/C1.jpg',
            p+'/asl_alphabet_test/D1.jpg'
          ]}/>

        <InferenceShowcase pictureUrls={[
            p+'/asl_alphabet_test/A_test.jpg',
            p+'/asl_alphabet_test/B_test.jpg',
            p+'/asl_alphabet_test/C_test.jpg',
            p+'/asl_alphabet_test/D_test.jpg',
            p+'/asl_alphabet_test/E_test.jpg',
            p+'/asl_alphabet_test/F_test.jpg',
            p+'/asl_alphabet_test/G_test.jpg',
            p+'/asl_alphabet_test/H_test.jpg'
          ]}/>
        <InferenceShowcase/>
      </ModelShowcase>
    </article>
  );
}

export default App;
