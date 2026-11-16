import { useState } from 'react';
import { ArrowUpRight, Check, Film, Plus, X } from 'lucide-react';
import type { Workbench } from '../lib/useWorkbench';
import { clipName } from '../lib/types';

export function Library({ w, close }: { w: Workbench; close: () => void }) {
  const [filter, setFilter] = useState<'footage' | 'synthetic'>('footage');
  const clips = w.clips.filter(c => filter === 'synthetic' ? c.synthetic : !c.synthetic);
  return <aside className="library" aria-label="Source library">
    <div className="rail-heading"><span className="eyebrow">SOURCE LIBRARY</span><button className="icon-button mobile-only" onClick={close} aria-label="Close source library"><X size={17}/></button><span className="counter">{String(clips.length).padStart(2, '0')}</span></div>
    <div className="segmented small" aria-label="Source type">{(['footage', 'synthetic'] as const).map(f => <button key={f} aria-pressed={filter === f} onClick={() => setFilter(f)}>{f === 'footage' ? 'Footage' : 'Synthetic'}</button>)}</div>
    <div className="source-list">
      {w.loading && <div className="library-loading" role="status">Connecting to your library…</div>}
      {clips.map((clip, i) => {
        const selected = w.clip?.id === clip.id || w.clip?.config?.source === clip.id;
        return <button key={clip.id} className={`source-item ${selected ? 'selected' : ''}`} disabled={w.locked} aria-pressed={selected} onClick={() => { w.chooseClip(clip); close(); }}>
          <div className="source-image"><img src={`/api/frames/${clip.id}/original/${Math.min(14, clip.count - 1)}`} alt={clipName(clip)}/><span className="source-number">{String(i + 1).padStart(2, '0')}</span>{selected && <span className="source-selected"><Check size={12}/></span>}<span className="duration">{(clip.count / clip.fps).toFixed(2)}s</span></div>
          <div className="source-caption"><strong>{clipName(clip)}</strong><span>{clip.synthetic ? 'Procedural study' : clip.dataset_sequence ? 'DAVIS / Natural footage' : 'Your local upload'}</span></div>
        </button>;
      })}
    </div>
    <label className={`import-button ${w.locked ? 'disabled' : ''}`}><Plus size={16}/> Import footage<input aria-label="Upload video or images" type="file" multiple accept="video/mp4,video/quicktime,image/png,image/jpeg" disabled={w.locked} onChange={e => { void w.upload(e.target.files); e.target.value = ''; }}/></label>
    <p className="fine-print">MP4, MOV or image sequence.<br/>Up to 8 seconds · 30 MiB.</p>
    {w.clip?.license && <a className="source-attribution" href={w.clip.source_url} target="_blank" rel="noreferrer">Footage attribution <ArrowUpRight size={12}/><span>DAVIS · {w.clip.license}</span></a>}
    <div className="library-history"><div className="rail-heading"><span className="eyebrow">THIS SESSION</span><span className="counter">{w.history.length}</span></div>{w.history.length ? w.history.map((job, i) => <button key={job.id} disabled={w.locked} onClick={() => w.openJob(job)}><Film size={14}/><span>{clipName(job.report?.source)}<small>Run {String(w.history.length - i).padStart(2, '0')} · {Object.keys(job.report?.methods ?? {}).length} methods</small></span><span className="history-dot"/></button>) : <p className="fine-print">Completed experiments will be kept here for this session.</p>}</div>
  </aside>;
}
