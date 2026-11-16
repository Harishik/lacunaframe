import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, ChevronLeft, ChevronRight, Maximize2, Pause, Play, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import type { Workbench } from '../lib/useWorkbench';
import { clipName, fmt, METHODS, timecode } from '../lib/types';
import type { Method, View } from '../lib/types';

function FrameImage({ src, alt, style }: { src: string; alt: string; style?: React.CSSProperties }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  return <><img key={src} src={src} alt={alt} style={{ ...style, opacity: loaded ? 1 : 0 }} onLoad={() => setLoaded(true)} onError={() => setFailed(true)} decoding="async"/>{!loaded && <span className="image-status">{failed ? 'Preview unavailable. Reselect the source; temporary runs expire after one hour.' : 'Loading frame…'}</span>}</>;
}
export function Viewer({ w, results }: { w: Workbench; results: () => void }) {
  const [split, setSplit] = useState(50);
  const [base, setBase] = useState('original');
  const [zoom, setZoom] = useState(false);
  const screen = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest('input,select,button,textarea,dialog') || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.code === 'Space') { event.preventDefault(); w.setPlaying(p => !p); }
      if (event.key === 'ArrowRight') { event.preventDefault(); w.setIndex(i => Math.min(w.count - 1, i + 1)); }
      if (event.key === 'ArrowLeft') { event.preventDefault(); w.setIndex(i => Math.max(0, i - 1)); }
      if (event.key === 'Home') { event.preventDefault(); w.setIndex(0); }
    };
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, [w]);
  const canView = (view: View) => view === 'original' || (['corrupted', 'mask'].includes(view) ? !!w.scenario : !!w.metric);
  const views: { id: View; name: string }[] = [{ id: 'original', name: 'Original' }, { id: 'corrupted', name: 'Damaged' }, { id: 'compare', name: 'Compare' }, { id: 'reconstructed', name: 'Recovered' }, { id: 'error', name: 'Error map' }, { id: 'mask', name: 'Mask' }];
  const missing = w.active?.missing?.[w.index];
  const indices = Array.from({ length: Math.min(9, w.count) }, (_, i) => Math.round(i * (w.count - 1) / (Math.min(9, w.count) - 1 || 1)));
  const imageStyle = { transform: zoom ? 'scale(2)' : 'scale(1)' };
  const kind = w.view === 'reconstructed' ? w.method : w.view;
  return <section className="viewer-column" aria-label="Frame viewer">
    <div className="viewer-title"><div><span className="eyebrow">SEQUENCE / {w.active?.synthetic ? 'SYNTHETIC' : 'NATURAL FOOTAGE'}</span><h1>{clipName(w.active)}</h1></div><span className="sequence-spec">320 × 180 <i/> {w.fps} FPS <i/> {w.count} FRAMES</span></div>
    <div className="viewer-toolbar"><div className="view-tabs">{views.map(v => <button key={v.id} aria-pressed={w.view === v.id} disabled={!canView(v.id)} onClick={() => w.setView(v.id)}>{v.name}</button>)}</div><div className="viewer-tools"><button className="icon-button" aria-label={zoom ? 'Zoom out' : 'Zoom in'} onClick={() => setZoom(!zoom)}>{zoom ? <ZoomOut size={15}/> : <ZoomIn size={15}/>}</button><button className="icon-button" aria-label="Fullscreen viewer" onClick={() => { if (document.fullscreenElement) void document.exitFullscreen(); else void screen.current?.requestFullscreen().catch(() => w.setError('Fullscreen is not available in this browser.')); }}><Maximize2 size={14}/></button></div></div>
    <div className="screen" ref={screen}>
      {!w.active ? <div className="screen-empty"><span className="empty-frame-mark"/><h2>{w.loading ? 'Opening the archive.' : 'Your footage belongs here.'}</h2><p>{w.loading ? 'Connecting to the local engine…' : 'Select a sequence or import a short clip.'}</p></div> : w.view === 'compare' && w.metric ? <>
        <FrameImage key={`${base}-${w.index}-${w.active.id}`} src={w.frame(base)} alt={base === 'original' ? 'Reference frame' : 'Corrupted frame'} style={imageStyle}/>
        <div className="comparison-layer" style={{ clipPath: `inset(0 0 0 ${split}%)` }}><FrameImage key={`${w.method}-${w.index}-${w.active.id}`} src={w.frame(w.method)} alt="Reconstructed frame" style={imageStyle}/></div>
        <div className="wipe-line" style={{ left: `${split}%` }}><span><ChevronLeft size={12}/><ChevronRight size={12}/></span></div>
        <input className="wipe-input" aria-label="Comparison divider" type="range" min="0" max="100" value={split} onChange={e => setSplit(Number(e.target.value))}/>
        <span className="screen-label left">{base === 'original' ? 'REFERENCE' : 'DAMAGED'}</span><span className="screen-label right"><i/>{METHODS[w.method].short.toUpperCase()} / RECOVERED</span>
      </> : <><FrameImage key={`${kind}-${w.index}-${w.active.id}`} src={w.frame(kind)} alt={`${w.view} frame ${w.index}`} style={imageStyle}/><span className="screen-label left">{views.find(v => v.id === w.view)?.name.toUpperCase()}</span></>}
      {w.active && <div className="screen-bottom"><span className={missing ? 'sample-state absent' : 'sample-state'}><i/>{missing ? w.metric && ['compare', 'reconstructed'].includes(w.view) ? 'INFERRED SAMPLE' : 'MISSING SAMPLE' : 'OBSERVED SAMPLE'}</span><span>{zoom ? '2× detail' : 'Native data / fitted view'} <b>{timecode(w.index, w.fps)}</b></span></div>}
    </div>
    <div className="viewer-context">{w.metric ? <><label><span>Method</span><select aria-label="Inspect method" value={w.method} onChange={e => w.setMethod(e.target.value as Method)}>{Object.keys(w.job!.report!.methods).map(m => <option key={m} value={m}>{METHODS[m as Method].name}</option>)}</select></label>{w.view === 'compare' && <label><span>Against</span><select aria-label="Comparison source" value={base} onChange={e => setBase(e.target.value)}><option value="original">Original reference</option><option value="corrupted">Damaged input</option></select></label>}<span className="viewer-hint">Drag divider to compare</span></> : <><span>{w.scenario ? `${w.scenario.missing?.filter(Boolean).length} of ${w.count} samples removed. Select methods to reconstruct.` : 'Untouched reference. Your experiment starts here.'}</span><span className="viewer-hint">{w.active?.synthetic ? 'Generated diagnostic' : 'Recorded footage'}</span></>}</div>
    <div className="timeline">
      <div className="transport"><div><button className="play-button" aria-label={w.playing ? 'Pause' : 'Play'} disabled={!w.active} onClick={() => w.setPlaying(!w.playing)}>{w.playing ? <Pause size={15} fill="currentColor"/> : <Play size={15} fill="currentColor"/>}</button><button className="icon-button" aria-label="Previous frame" onClick={() => w.setIndex(i => Math.max(0, i - 1))}><ChevronLeft size={16}/></button><button className="icon-button" aria-label="Next frame" onClick={() => w.setIndex(i => Math.min(w.count - 1, i + 1))}><ChevronRight size={16}/></button><button className="icon-button" aria-label="Rewind" onClick={() => w.setIndex(0)}><RotateCcw size={13}/></button><span className="timecode">{timecode(w.index, w.fps)} <span>/ {timecode(w.count, w.fps)}</span></span></div><span className="frame-id">FRAME {String(w.index).padStart(3, '0')} <span>OF {String(w.count - 1).padStart(3, '0')}</span></span></div>
      <div className="filmstrip">{indices.map(i => <button key={i} aria-label={`Go to frame ${i}`} aria-current={w.index === i ? 'true' : undefined} onClick={() => w.setIndex(i)}>{w.active && <img src={w.frame(w.metric ? w.method : w.scenario ? 'corrupted' : 'original', i)} alt=""/>}<span>{String(i).padStart(3, '0')}</span></button>)}<div className="playhead" style={{ left: `${w.index / Math.max(1, w.count - 1) * 100}%` }}><i/></div></div>
      <input className="timeline-scrubber" aria-label="Timeline" type="range" min="0" max={w.count - 1} value={w.index} onChange={e => w.setIndex(Number(e.target.value))}/>
      <div className="sample-track" aria-label="Sample availability">{Array.from({ length: w.count }, (_, i) => <button key={i} aria-label={`Frame ${i}${w.active?.missing?.[i] ? ' missing' : ' observed'}`} className={`${w.active?.missing?.[i] ? 'missing' : ''} ${w.index === i ? 'current' : ''}`} onClick={() => w.setIndex(i)}/>)}</div>
      <div className="timeline-legend"><span><i/> Observed <i className="damage-dot"/> Missing</span><span><kbd>SPACE</kbd> Play <kbd>← →</kbd> Step</span></div>
    </div>
    {w.metric ? <div className="run-summary"><div className="summary-heading"><span><Check size={13}/> RECONSTRUCTION COMPLETE</span><button className="text-button" onClick={results}>Inspect results <ArrowRight size={14}/></button></div><div className="summary-metrics"><div><span>Affected PSNR ↑</span><strong>{fmt(w.metric.affected_psnr)}<small>dB</small></strong></div><div><span>Affected SSIM ↑</span><strong>{fmt(w.metric.affected_ssim, 3)}</strong></div><div><span>Temporal error ↓</span><strong>{fmt(w.metric.temporal_error, 4)}</strong></div><div><span>Inference time</span><strong>{fmt(w.metric.runtime_seconds, 2)}<small>s</small></strong></div></div><p className="summary-footnote">{missing ? `Frame ${w.index} was absent from the input. This estimate is not a verified observation.` : 'This frame was observed. Temporal methods preserve its pixels unchanged.'}</p></div> : <div className="workflow-note"><span className="workflow-number">{w.scenario ? '02' : '01'}</span><div><strong>{w.scenario ? 'An absence, ready to investigate.' : 'A controlled experiment in missing time.'}</strong><p>{w.scenario ? 'Choose your reconstruction methods in the inspector. Every method receives the same damaged input.' : 'Choose a damage pattern in the inspector, then compare what each method can recover.'}</p></div><ArrowRight size={18}/></div>}
  </section>;
}
