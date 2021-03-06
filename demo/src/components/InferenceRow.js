import { CloseCircleOutlined, DownOutlined, RightOutlined } from '@ant-design/icons';
import { Button, Empty, List, Row, Tooltip } from 'antd';
import loadImage from 'blueimp-load-image';
import React, { useEffect, useRef, useState } from 'react';
import { infer } from '../utils/inference';
import { InferenceResults } from './InferenceResults';

export function InferenceRow(props) {
  const initialInfResult = {
    time: -1,
    probabilities: [],
    prediction: null,
    loading: false
  };
  const [inferenceResult, setInferenceResult] = useState(initialInfResult);
  const { imgSize } = props.model;
  const canvasElement = useRef(null);
  const [collapsed, setCollapsed] = useState(true);

  // draw image to canvas
  async function drawimg(crop) {
    const blueimg = await loadImage(props.picture.base64data, {
      maxWidth: props.model.imgSize,
      crop: true,
      canvas: true,
      cover: true
    })
    if (!canvasElement.current) return console.warn('No canvas (drawimg)');
    const ctx = canvasElement.current.getContext('2d');
    if(crop) {
      // resize to 256x256 so it can be center cropped
      console.log("crop!")
      ctx.drawImage(blueimg.image, -16, -16, 256, 256)
    } else {
      ctx.drawImage(blueimg.image, 0, 0);
    }
    // setImageLoaded(true);
  }

  async function inferimg() {
    setInferenceResult({ ...initialInfResult, loading: true });
    const { session, model } = props;

    // inference
    if (!canvasElement.current) return console.warn('No canvas (inferimg)');

    const tensor = model.tensor(canvasElement.current);
    const result = await infer(model, session, tensor);
    console.log('inference result', result);

    // wait 500ms before showing result
    setTimeout(() => {
      setInferenceResult({ ...result, loading: false });
    }, 750)
  }

  useEffect(() => { // Preprocess image
    if (!props.picture.base64data) return;

    drawimg(props.crop)
    //.then(() => props.session && inferimg());
  }, [props.picture.base64data, props.model.imgSize, props.session]);

  const RemoveButton = () => (
    <Tooltip title='Remove picture'>
      <Button onClick={() => props.onRemove()} type='text'
        icon={<CloseCircleOutlined />} />
    </Tooltip>
  );

  const { loading, time, probabilities, prediction } = inferenceResult;
  const InferenceButton = () => {
    const canInfere = !props.session || !props.picture.base64data;
    let tooltip = 'Perform inference';
    if (!props.session) tooltip = 'No model session available';
    if (!props.picture.base64data) tooltip = 'No image loaded';
    return (
      <>
        <Row>
          <Tooltip title={tooltip}>
            <Button onClick={() => inferimg()} loading={loading}
              disabled={canInfere} >
              Inference
            </Button>
          </Tooltip>
        </Row>
        <Row>
          <small style={{ color: '#ccc' }}>
            {time !== -1 ? `Inference took ${time}ms` : <>&nbsp;</>}
          </small>
        </Row>
      </>
    )
  };

  const CollapseButton = () => {
    if (collapsed)
      return <Button onClick={() => setCollapsed(false)} type='text'
        icon={<RightOutlined />} />
    else
      return <Button onClick={() => setCollapsed(true)} type='text'
        icon={<DownOutlined />} />
  }

  return (
    <List.Item actions={[<RemoveButton />, <InferenceButton />]}
      className='App-picitem'>
      <List.Item.Meta title={props.picture.file.name.replace('_', '-')}
        description={`${imgSize} x ${imgSize}`}
        avatar={props.picture.base64data ?
          <canvas ref={canvasElement} width={imgSize} height={imgSize} 
          style={{ minWidth: 50, maxWidth: 140 }}/> :
          <Empty description='Image could not be loaded'
            style={{ margin: '20px' }} />}
      />

      <div className='ant-list-item-collapse'>
        <CollapseButton />
      </div>
      <InferenceResults probabilities={probabilities} prediction={prediction}
        top_n={collapsed ? 3 : 10} />
    </List.Item>
  );
}