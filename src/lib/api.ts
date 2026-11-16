export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api${path}`, {
    signal,
    ...(body ? {
      method: 'POST',
      headers: body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
      body: body instanceof FormData ? body : JSON.stringify(body),
    } : {}),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: 'The processing engine could not be reached.' }));
    throw new Error(typeof data.detail === 'string' ? data.detail : 'These settings could not be processed. Check the input and try again.');
  }
  return response.json() as Promise<T>;
}
export const message = (error: unknown) => error instanceof Error ? error.message : String(error);
