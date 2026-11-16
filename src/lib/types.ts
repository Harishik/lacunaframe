export type Method = 'hold' | 'linear' | 'flow' | 'telea' | 'rife' | 'rife_fast';
export type View = 'original' | 'corrupted' | 'compare' | 'reconstructed' | 'mask' | 'error';
export type Page = 'studio' | 'evaluation' | 'project';
export interface Config {
  seed: number; drop: number; gap: number; start: number;
  occlusion: number; mask_shape: 'rectangle' | 'brush'; noise: number; blur: number;
}
export interface Clip {
  id: string; name: string; count: number; fps: number; synthetic: boolean;
  missing?: boolean[]; license?: string; attribution?: string; source_url?: string;
  dataset_sequence?: string; config?: Config & { source: string };
}
export interface Capability { model: { available: boolean; name: string; provider: string; threads: number } }
export interface Row { frame: number; time: number; mse: number; psnr: number | string; ssim: number | null }
export interface Metric {
  psnr: number | string; ssim: number | null; affected_psnr: number | string | null;
  affected_ssim: number | null; temporal_error: number; runtime_seconds: number;
  process_peak_rss_mib: number; model_initialization_seconds: number; frames: Row[];
  warnings: string[]; uncertainty: number[] | null; media: string;
}
export interface Report { methods: Partial<Record<Method, Metric>>; corrupted: Metric; source: Clip }
export interface Job { id: string; status: 'running' | 'complete' | 'failed'; progress: number; stage?: string; report?: Report; error?: string }
export interface Benchmark {
  protocol: string;
  summary: { regime: string; method: Method; n: number; mean_affected_psnr: number; mean_affected_ssim: number; mean_seconds_per_missing_frame: number }[];
}
export const METHODS: Record<Method, { name: string; short: string; description: string; family: string }> = {
  hold: { name: 'Nearest frame', short: 'Hold', description: 'Reuse the closest observed frame.', family: 'Classical' },
  linear: { name: 'Linear interpolation', short: 'Linear', description: 'Blend the two observed neighbors.', family: 'Classical' },
  flow: { name: 'Optical flow', short: 'Flow', description: 'Align motion with bidirectional warps.', family: 'Classical' },
  rife: { name: 'RIFE 4.9', short: 'RIFE', description: 'Pretrained neural interpolation on CPU.', family: 'Learned' },
  rife_fast: { name: 'RIFE · fast', short: 'RIFE fast', description: 'Half-resolution inference. Less detail.', family: 'Learned' },
  telea: { name: 'Telea inpainting', short: 'Telea', description: 'Repair spatial masks; hold absent frames.', family: 'Spatial' },
};
export const DEFAULT_CONFIG: Config = { seed: 42, drop: .15, gap: 3, start: .4, occlusion: 0, mask_shape: 'rectangle', noise: 0, blur: 0 };
export const fmt = (value: number | string | null | undefined, digits = 2): string =>
  value === null || value === undefined ? '—' : value === 'Infinity' ? '∞' : typeof value === 'number' ? value.toFixed(digits) : value;
export const clipName = (clip?: Clip | null): string => clip?.dataset_sequence ? ({ blackswan: 'Black swan', 'bmx-trees': 'Between the trees', breakdance: 'Body in motion' }[clip.dataset_sequence] ?? clip.name) : clip?.name ?? 'Choose a sequence';
export const timecode = (frame: number, fps: number): string => `${Math.floor(frame / fps).toString().padStart(2, '0')}:${(frame % fps).toString().padStart(2, '0')}`;
