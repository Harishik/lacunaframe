import { useEffect, useRef, useState } from 'react';
import { AlertCircle, ArrowUpRight, FolderOpen, Keyboard, X } from 'lucide-react';
import { useWorkbench } from './lib/useWorkbench';
import type { Benchmark, Page } from './lib/types';
import { Library } from './components/Library';
import { Inspector } from './components/Inspector';
import { Viewer } from './components/Viewer';
import { Evaluation } from './components/Evaluation';
import { Project } from './components/Project';

function currentPage(): Page {
  const value = location.hash.slice(1);
  return value === 'evaluation' ? 'evaluation' : ['project', 'question', 'approach', 'evidence', 'limits'].includes(value) ? 'project' : 'studio';
}
function Mark() {
  return <svg width="24" height="27" viewBox="0 0 24 27" fill="none" aria-hidden="true"><path d="M2 0v27h6V6h5V0H2Zm14 0v9h6V0h-6ZM11 12v6h5v9h6V12H11Z" fill="currentColor"/></svg>;
}
export default function App() {
  const w = useWorkbench();
  const [page, setPage] = useState<Page>(currentPage);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const help = useRef<HTMLDialogElement>(null);
  useEffect(() => { const listener = () => setPage(currentPage()); window.addEventListener('hashchange', listener); return () => window.removeEventListener('hashchange', listener); }, []);
  useEffect(() => { fetch('/reports/natural-study.json').then(r => { if (!r.ok) throw new Error('Unavailable'); return r.json(); }).then(setBenchmark).catch(() => {}); }, []);
  const navigate = (next: Page) => { w.setPlaying(false); location.hash = next; setPage(next); window.scrollTo({ top: 0 }); };
  return <div className={`application page-${page}`}>
    <a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to content</a>
    <header className="app-header"><a className="wordmark" href="#studio" aria-label="Lacunaframe workspace" onClick={() => navigate('studio')}><Mark/><span>LACUNAFRAME<small>THE RECONSTRUCTION DESK</small></span></a><nav aria-label="Main navigation">{([{ id: 'studio', label: 'Workspace' }, { id: 'evaluation', label: 'Evaluation' }, { id: 'project', label: 'The project' }] as const).map(item => <button key={item.id} aria-current={page === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}>{item.label}{item.id === 'project' && <ArrowUpRight size={12}/>}</button>)}</nav><div className="header-status"><span><i className={w.capability ? 'status-dot' : 'status-dot offline'}/>{w.capability ? 'LOCAL ENGINE' : 'ENGINE OFFLINE'}</span><button className="icon-button" aria-label="Keyboard shortcuts" onClick={() => help.current?.showModal()}><Keyboard size={17}/></button><span className="release">0.3</span></div></header>
    {w.error && page !== 'project' && <div className="error-banner" role="alert"><AlertCircle size={17}/><span>{w.error}</span>{!w.clip && <button onClick={w.retry}>Retry connection</button>}<button className="icon-button" aria-label="Dismiss error" onClick={() => w.setError('')}><X size={15}/></button></div>}
    <div id="main-content" tabIndex={-1}>
      {page === 'studio' ? <><div className="workspace-topline"><span><i/> UNTITLED SESSION <span className="topline-separator">/</span> Video reconstruction</span><button className="text-button mobile-only" onClick={() => setLibraryOpen(!libraryOpen)}><FolderOpen size={15}/> Source library</button><span className="topline-right">Every missing frame leaves a trace.</span></div><main className={`studio-layout ${libraryOpen ? 'library-open' : ''}`}><Library w={w} close={() => setLibraryOpen(false)}/><Viewer w={w} results={() => navigate('evaluation')}/><Inspector w={w}/></main></> : page === 'evaluation' ? <Evaluation w={w} benchmark={benchmark} studio={() => navigate('studio')}/> : <Project studio={() => navigate('studio')} evaluation={() => navigate('evaluation')}/>}
    </div>
    <footer className="app-footer"><span><Mark/> INDEPENDENT RESEARCH / EDUCATIONAL PROTOTYPE</span><span>{page === 'studio' ? 'Uploads stay local. Temporary experiments expire after one hour.' : 'Not affiliated with Korea University or the PRISM Lab.'}</span><span>LACUNAFRAME © 2026</span></footer>
    <dialog aria-label="Keyboard shortcuts" ref={help} className="shortcut-dialog"><div><span className="eyebrow">AT THE EDITING DESK</span><button className="icon-button" aria-label="Close shortcuts" onClick={() => help.current?.close()}><X size={18}/></button></div><h2>A few useful keys.</h2><dl><dt>Play / pause</dt><dd><kbd>SPACE</kbd></dd><dt>Previous / next frame</dt><dd><kbd>←</kbd> <kbd>→</kbd></dd><dt>First frame</dt><dd><kbd>HOME</kbd></dd><dt>Close this panel</dt><dd><kbd>ESC</kbd></dd></dl><p>Viewer shortcuts work in the workspace when focus is outside a form control. Use Tab to move between controls and arrow keys to adjust sliders.</p></dialog>
  </div>;
}
