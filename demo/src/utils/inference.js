export async function infer(model, session, tensor) {
    const start = new Date();
    const outputData = session.predict(tensor);
    const end = new Date();
    const time = (end.getTime() - start.getTime());
    const output = outputData.gather(0);
    const { probabilities, prediction } = model.postprocess(output);
    return { time, probabilities, prediction }
}