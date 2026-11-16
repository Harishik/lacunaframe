import { useEffect, useId, useState } from 'react';
import { ArrowRight, Check, ChevronDown, LoaderCircle, RotateCcw, SlidersHorizontal } from 'lucide-react';
import type { Workbench } from '../lib/useWorkbench';
import { DEFAULT_CONFIG, METHODS } from '../lib/types';
import type { Config, Method } from '../lib/types';

function Slider({ label, value, max, step, unit, disabled, onChange }: { label: string; value: number; max: number; step: number; unit?: string; disabled: boolean; onChange: (v: number) => void }) {
  const id = useId();
  return <label htmlFor={id} className="slider-field"><span>{label}<output>{unit === '%' ? Math.round(value * 100) : value}{unit === '%' ? '%' : unit ?? ''}</output></span><input id={id} type="range" min="0" max={max} step={step} value={value} disabled={disabled} onChange={e => onChange(Number(e.target.value))}/></label>;
}
export function Inspector({ w }: { w: Workbench }) {
  const [tab, setTab] = useState<'damage' | 'restore'>('damage');
  const [advanced, setAdvanced] = useState(false);
  const [preset, setPreset] = useState('custom');
  useEffect(() => { if (!w.scenario) setTab('damage'); }, [w.scenario]);
  const apply = (name: string, values: Partial<Config>) => { setPreset(name); w.changeConfig({ ...DEFAULT_CONFIG, drop: 0, gap: 0, ...values }); };
  const change = (values: Partial<Config>) => { setPreset('custom'); w.changeConfig(values); };
  const busy = w.job?.status === 'running';
  return <aside className="inspector" aria-label="Experiment settings">
    <div className="rail-heading"><span className="eyebrow">EXPERIMENT SETUP</span><SlidersHorizontal size={14}/></div>
    <div className="inspector-tabs"><button aria-pressed={tab === 'damage'} onClick={() => setTab('damage')}><span>01</span> Damage</button><button aria-pressed={tab === 'restore'} onClick={() => setTab('restore')}><span>02</span> Restore{w.scenario && <i/>}</button></div>
    <div className="inspector-body">
      {tab === 'damage' ? <>
        <h2>Make an absence.</h2><p className="body-note">Remove observations. Keep the original untouched for comparison.</p>
        <div className="preset-label">START WITH A PATTERN</div>
        <div className="presets">
          <button className={preset === 'scatter' ? 'active' : ''} disabled={w.locked} onClick={() => apply('scatter', { drop: .2 })}><span className="pattern scattered"><i/><i/><i/><i/><i/><i/><i/></span>Scattered</button>
          <button className={preset === 'gap' ? 'active' : ''} disabled={w.locked} onClick={() => apply('gap', { gap: 6 })}><span className="pattern gap"><i/><i/><i/><i/><i/><i/><i/></span>Interval</button>
          <button className={preset === 'mask' ? 'active' : ''} disabled={w.locked} onClick={() => { apply('mask', { occlusion: .25 }); setAdvanced(true); }}><span className="pattern occluded"><i/></span>Occlusion</button>
        </div>
        <div className="control-group">
          <Slider label="Dropped frames" value={w.config.drop} max={.8} step={.05} unit="%" disabled={w.locked} onChange={drop => change({ drop })}/>
          <Slider label="Missing interval" value={w.config.gap} max={Math.min(24, w.count - 1)} step={1} unit=" frames" disabled={w.locked} onChange={gap => change({ gap })}/>
          <Slider label="Interval starts at" value={w.config.start} max={1} step={.05} unit="%" disabled={w.locked} onChange={start => change({ start })}/>
        </div>
        <button className="disclosure" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>Spatial damage & noise<ChevronDown size={14} className={advanced ? 'rotated' : ''}/></button>
        {advanced && <div className="advanced-controls">
          <Slider label="Occlusion size" value={w.config.occlusion} max={.7} step={.05} unit="%" disabled={w.locked} onChange={occlusion => change({ occlusion })}/>
          <label className="select-field">Mask shape<select value={w.config.mask_shape} disabled={w.locked} onChange={e => change({ mask_shape: e.target.value as Config['mask_shape'] })}><option value="rectangle">Rectangle</option><option value="brush">Brush stroke</option></select></label>
          <Slider label="Noise strength σ" value={w.config.noise} max={60} step={2} disabled={w.locked} onChange={noise => change({ noise })}/>
          <Slider label="Blur σ" value={w.config.blur} max={3} step={.25} disabled={w.locked} onChange={blur => change({ blur })}/>
          <p className="fine-print">Noise and blur are stress tests. No denoising or deblurring solver is included.</p>
        </div>}
        <div className="seed-field"><label htmlFor="seed">Random seed <small>Repeatable by design</small></label><input id="seed" type="number" min="0" max="4294967295" value={w.config.seed} disabled={w.locked} onChange={e => change({ seed: Math.max(0, Math.min(4294967295, Number(e.target.value))) })}/></div>
        <button className="text-button reset" disabled={w.locked} onClick={() => { setPreset('custom'); w.changeConfig(DEFAULT_CONFIG); }}><RotateCcw size={12}/> Reset settings</button>
      </> : <>
        <h2>Find what remains.</h2><p className="body-note">Run several methods on the same damaged sequence.</p>
        {!w.scenario && <p className="inline-notice">Create a damaged clip in step 01 first.</p>}
        <div className="method-options">{(Object.keys(METHODS) as Method[]).map(m => <label key={m} className={w.methods.includes(m) ? 'checked' : ''}><input type="checkbox" checked={w.methods.includes(m)} disabled={w.locked || (m.startsWith('rife') && !w.capability?.model.available)} onChange={() => w.setMethods(ms => ms.includes(m) ? ms.filter(v => v !== m) : [...ms, m])}/><span className="method-check">{w.methods.includes(m) && <Check size={11}/>}</span><span className="method-info"><strong>{METHODS[m].name}<small>{METHODS[m].family}</small></strong><span>{METHODS[m].description}</span></span></label>)}</div>
        <div className="engine-note"><span className="status-dot"/> {w.capability?.model.available ? `RIFE ready · CPU · ${w.capability.model.threads} threads` : 'Classical methods available'}<p>Observed pixels stay intact.<br/>Reference frames never enter the solvers.</p></div>
      </>}
    </div>
    <div className="inspector-action">
      {busy && <div className="processing" role="status"><div><span>{w.job?.stage ?? 'Preparing'}</span><b>{Math.round((w.job?.progress ?? 0) * 100)}%</b></div><progress max="1" value={w.job?.progress ?? 0}/></div>}
      {tab === 'damage' ? <button className="button primary" disabled={!w.clip || w.locked} onClick={async () => { if (await w.damage()) setTab('restore'); }}>{w.busy ? <LoaderCircle size={15} className="spin"/> : null}{w.busy ? 'Preparing…' : 'Create damaged clip'}<ArrowRight size={16}/></button> : <button className="button primary" disabled={!w.scenario || !w.methods.length || w.locked} onClick={() => void w.run()}>{busy ? <LoaderCircle size={15} className="spin"/> : null}{busy ? 'Reconstructing…' : `Run ${w.methods.length} ${w.methods.length === 1 ? 'method' : 'methods'}`}<ArrowRight size={16}/></button>}
      <span className="action-caption">{tab === 'damage' ? 'Original reference is preserved' : 'Local processing. No external inference.'}</span>
    </div>
  </aside>;
}
