import { useCallback, useEffect, useState } from 'react';
import { api, message } from './api';
import { DEFAULT_CONFIG } from './types';
import type { Capability, Clip, Config, Job, Method, View } from './types';

export function useWorkbench() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [clip, setClip] = useState<Clip | null>(null);
  const [capability, setCapability] = useState<Capability | null>(null);
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [scenario, setScenario] = useState<Clip | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [methods, setMethods] = useState<Method[]>(['linear', 'flow', 'rife']);
  const [method, setMethod] = useState<Method>('rife');
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>('original');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [connection, setConnection] = useState(0);
  const active = job?.report?.source ?? scenario ?? clip;
  const count = active?.count ?? 1;
  const fps = active?.fps ?? 12;
  const locked = busy || job?.status === 'running';
  const metric = job?.report?.methods[method];

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError('');
    Promise.all([api<Clip[]>('/demos', undefined, controller.signal), api<Capability>('/capabilities', undefined, controller.signal)])
      .then(([sources, engine]) => {
        setClips(sources); setClip(sources[0]); setCapability(engine);
        if (!engine.model.available) { setMethods(['linear', 'flow']); setMethod('flow'); }
      }).catch(e => { if (!controller.signal.aborted) setError(message(e)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [connection]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setIndex(i => (i + 1) % count), 1000 / fps);
    return () => clearInterval(timer);
  }, [playing, count, fps]);

  useEffect(() => {
    if (job?.status !== 'running') return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api<Job>(`/results/${job.id}`);
        if (stopped) return;
        setJob(next);
        if (next.status === 'complete') {
          setHistory(h => [next, ...h.filter(j => j.id !== next.id)].slice(0, 10));
          setIndex(Math.max(0, next.report?.source.missing?.findIndex(Boolean) ?? 0));
          setView('compare');
        } else if (next.status === 'failed') setError(next.error ?? 'The experiment failed. Try again.');
        else timer = setTimeout(poll, 450);
      } catch (e) {
        if (!stopped) { setError(message(e)); setJob(null); }
      }
    };
    timer = setTimeout(poll, 250);
    return () => { stopped = true; clearTimeout(timer); };
  }, [job?.id, job?.status]);

  const invalidate = useCallback(() => {
    setScenario(null); setJob(null); setIndex(0); setPlaying(false); setView('original');
  }, []);
  const chooseClip = (source: Clip) => { invalidate(); setClip(source); setError(''); };
  const changeConfig = (next: Partial<Config>) => { invalidate(); setConfig(c => ({ ...c, ...next })); };
  const damage = async () => {
    if (!clip || locked) return false;
    setBusy(true); setPlaying(false); setError(''); setJob(null);
    try {
      const result = await api<Clip>('/scenarios', { source: clip.id, ...config });
      setScenario(result); setView('corrupted');
      setIndex(Math.max(0, result.missing?.findIndex(Boolean) ?? 0));
      return true;
    } catch (e) { setError(message(e)); return false; }
    finally { setBusy(false); }
  };
  const run = async () => {
    if (!scenario || locked || !methods.length) return;
    setBusy(true); setPlaying(false); setError('');
    try {
      setMethod(methods.includes('rife') ? 'rife' : methods[0]);
      setJob(await api<Job>('/run', { scenario: scenario.id, methods }));
      setView('corrupted');
    } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  };
  const upload = async (files: FileList | null) => {
    if (!files?.length || locked) return;
    setBusy(true); setError('');
    try {
      const data = new FormData(); Array.from(files).forEach(file => data.append('files', file));
      const result = await api<Clip>('/upload', data);
      setClips(c => [...c, result]); chooseClip(result);
    } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  };
  const openJob = (entry: Job) => {
    if (locked || !entry.report) return;
    setJob(entry); setScenario(entry.report.source); setClip(entry.report.source);
    if (entry.report.source.config) {
      const { source: _source, ...settings } = entry.report.source.config;
      void _source; setConfig(settings);
    }
    setMethod(Object.keys(entry.report.methods)[0] as Method);
    setIndex(Math.max(0, entry.report.source.missing?.findIndex(Boolean) ?? 0)); setPlaying(false); setView('compare');
  };
  const frame = (kind: string, at = index) => `/api/frames/${job?.report ? job.id : (scenario ?? clip)?.id}/${kind}/${at}?method=${method}`;
  return { clips, clip, capability, config, scenario, job, history, methods, method, index, playing, view, busy, loading, error, active, count, fps, locked, metric,
    chooseClip, changeConfig, damage, run, upload, openJob, frame, setMethods, setMethod, setIndex, setPlaying, setView, setError, retry: () => setConnection(c => c + 1) };
}
export type Workbench = ReturnType<typeof useWorkbench>;
