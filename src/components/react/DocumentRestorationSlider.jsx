import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';

export default function DocumentRestorationSlider() {
  return (
    <section className="surface-card h-full">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-cyan-50">Computer Vision Document Restoration</h3>
        <span className="metric-badge text-cyan-100">Generative cleanup pipeline</span>
      </div>

      <p className="mb-4 text-sm text-cyan-100/85">
        Slide to compare synthetic watermarked sheet music versus reconstructed output. This slot is wired for
        the future generative restoration model.
      </p>

      <div className="overflow-hidden rounded-xl border border-cyan-300/25">
        <ReactCompareSlider
          itemOne={
            <ReactCompareSliderImage
              src="/placeholders/sheet-music-watermarked.svg"
              alt="Watermarked sheet music"
            />
          }
          itemTwo={
            <ReactCompareSliderImage src="/placeholders/sheet-music-clean.svg" alt="Restored sheet music" />
          }
          position={35}
        />
      </div>
    </section>
  );
}
