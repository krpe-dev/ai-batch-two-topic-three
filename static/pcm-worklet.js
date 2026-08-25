class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();

    this.targetSampleRate = 16000;
    this.frameSize = 320; // 20 ms at 16 kHz
    this.outputBuffer = [];

    this.phase = 0;
    this.ratio = this.targetSampleRate / sampleRate;
  }

  process(inputs) {
    const input = inputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const channel = input[0];

    for (let i = 0; i < channel.length; i++) {
      this.phase += this.ratio;

      if (this.phase >= 1.0) {
        this.phase -= 1.0;

        let sample = channel[i];
        sample = Math.max(-1, Math.min(1, sample));

        const int16Sample = sample < 0
          ? sample * 0x8000
          : sample * 0x7fff;

        this.outputBuffer.push(int16Sample);

        if (this.outputBuffer.length === this.frameSize) {
          const pcmFrame = new Int16Array(this.outputBuffer);
          this.port.postMessage(pcmFrame.buffer, [pcmFrame.buffer]);
          this.outputBuffer = [];
        }
      }
    }

    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);