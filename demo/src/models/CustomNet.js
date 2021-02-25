import labels from './CustomNet_labels';
import * as tf from '@tensorflow/tfjs';

const imgSize = 64;
export default {
  imgSize,

  tensor(img) {
    const tfImg = tf.browser.fromPixels(img);
    const smalImg = tf.image.resizeBilinear(tfImg, [64, 64]);
    const resized = tf.cast(smalImg, 'float32');
    let t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,64,64,3])

    // standardize
    const mean = t4d.mean().dataSync();
    t4d = t4d.sub(mean);                      // 0 mean
    t4d.sub(mean).pow(2).div(t4d.size).sqrt() // divided by std dev.

    return t4d;
  },

  postprocess(output) {
    const probs = output.dataSync();
    console.log('probs', probs);
    const prediction=output.argMax().dataSync()[0];
    console.log('prediction', labels[prediction]);
    let probabilities = [];
    for (let i = 0; i < probs.length; i++) {
      probabilities.push({
        probability: probs[i],
        label: labels[i]
      })
    }
    return { probabilities, prediction };
  }
}